from typing import TypedDict, Annotated
import operator
import httpx
import json

from langgraph.graph import StateGraph, END
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.reranking.bge_reranker import BGEReranker


class AgentState(TypedDict):
    question:        str
    thoughts:        Annotated[list[str], operator.add]
    tool_calls:      Annotated[list[dict], operator.add]
    search_results:  Annotated[list[str], operator.add]
    final_answer:    str
    iterations:      int


class ReActAgent:
    """ReAct (Reasoning + Acting) agent for multi-hop research queries.
    
    The agent alternates between:
      Thought — reasoning about what information is needed
      Action  — calling a tool (search_corpus or finish)
      Observation — reading the tool result
    
    This loop continues until the agent has enough information to answer
    or reaches the maximum iteration limit.
    
    Unlike a fixed RAG pipeline, the agent can:
    - Decide whether retrieval is needed at all
    - Issue multiple targeted searches for different aspects
    - Refine its search based on what it found
    - Synthesize information across multiple retrievals
    """

    def __init__(self, retriever: HybridRetriever,
                 reranker: BGEReranker,
                 ollama_model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434",
                 max_iterations: int = 4):
        self.retriever  = retriever
        self.reranker   = reranker
        self.model      = ollama_model
        self.base_url   = base_url
        self.max_iter   = max_iterations
        self.graph      = self._build_graph()

    def _llm(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        return resp.json()["response"].strip()

    def _search(self, query: str, top_k: int = 5) -> str:
        results  = self.retriever.retrieve(query, top_k=top_k * 2)
        reranked = self.reranker.rerank(query, results, top_k=top_k)
        if not reranked:
            return "No relevant results found."
        parts = []
        for i, r in enumerate(reranked, 1):
            title = r.chunk.metadata.get("title", "Unknown")
            parts.append(f"[{i}] {title}\n{r.chunk.text[:400]}")
        return "\n\n".join(parts)

    def _reason_node(self, state: AgentState) -> AgentState:
        context = ""
        if state["search_results"]:
            context = "\n\n".join(state["search_results"][-3:])
            context = f"\nPrevious search results:\n{context}\n"

        history = ""
        for t, tc in zip(state["thoughts"], state["tool_calls"]):
            history += f"Thought: {t}\nAction: {tc}\n"

        prompt = f"""You are a research assistant answering questions about AI/ML papers.
{context}
Question: {state['question']}
{history}
What should you do next? Choose one:
1. Search for more information: respond with
   Thought: <your reasoning>
   Action: search(<your search query>)

2. You have enough information to answer: respond with
   Thought: <your reasoning>
   Action: finish(<your complete answer>)

Respond with exactly one Thought and one Action."""

        response = self._llm(prompt)

        thought, action = "", {}
        for line in response.split("\n"):
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action: search("):
                query = line.replace("Action: search(", "").rstrip(")")
                action = {"type": "search", "query": query}
            elif line.startswith("Action: finish("):
                answer = line.replace("Action: finish(", "").rstrip(")")
                action = {"type": "finish", "answer": answer}

        if not action:
            action = {"type": "finish", "answer": response}

        return {
            "thoughts":   [thought],
            "tool_calls": [action],
            "iterations": state["iterations"] + 1,
        }

    def _act_node(self, state: AgentState) -> AgentState:
        last_action = state["tool_calls"][-1]

        if last_action["type"] == "search":
            result = self._search(last_action["query"])
            return {"search_results": [result]}

        if last_action["type"] == "finish":
            return {"final_answer": last_action["answer"]}

        return {"final_answer": "Could not determine an answer."}

    def _should_continue(self, state: AgentState) -> str:
        if state.get("final_answer"):
            return "end"
        if state["iterations"] >= self.max_iter:
            return "end"
        last_action = state["tool_calls"][-1] if state["tool_calls"] else {}
        if last_action.get("type") == "finish":
            return "end"
        return "reason"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("reason", self._reason_node)
        graph.add_node("act",    self._act_node)
        graph.add_edge("reason", "act")
        graph.add_conditional_edges(
            "act",
            self._should_continue,
            {"reason": "reason", "end": END},
        )
        graph.set_entry_point("reason")
        return graph.compile()

    def run(self, question: str) -> dict:
        initial_state = AgentState(
            question=question,
            thoughts=[],
            tool_calls=[],
            search_results=[],
            final_answer="",
            iterations=0,
        )
        final_state = self.graph.invoke(initial_state)

        if not final_state["final_answer"]:
            results  = self.retriever.retrieve(question, top_k=10)
            reranked = self.reranker.rerank(question, results, top_k=5)
            context  = "\n\n".join(r.chunk.text for r in reranked)
            final_state["final_answer"] = self._llm(
                f"Based on this context:\n{context}\n\nAnswer: {question}"
            )

        return {
            "question":     question,
            "answer":       final_state["final_answer"],
            "thoughts":     final_state["thoughts"],
            "tool_calls":   final_state["tool_calls"],
            "iterations":   final_state["iterations"],
            "searches":     len(final_state["search_results"]),
        }