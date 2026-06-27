import httpx
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.retrieval.multi_query_retriever import MultiQueryRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator


class ResearchAgent:
    """Specialized sub-agent with a focused role."""

    def __init__(self, name: str, role: str, retriever,
                 reranker: BGEReranker,
                 ollama_model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434"):
        self.name      = name
        self.role      = role
        self.retriever = retriever
        self.reranker  = reranker
        self.model     = ollama_model
        self.base_url  = base_url

    def _llm(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        return resp.json()["response"].strip()

    def execute(self, task: str) -> dict:
        candidates = self.retriever.retrieve(task, top_k=10)
        reranked   = self.reranker.rerank(task, candidates, top_k=4)
        context    = "\n\n".join(r.chunk.text for r in reranked)
        sources    = [r.chunk.metadata.get("title", "") for r in reranked]

        prompt = (
            f"You are a {self.role}.\n\n"
            f"Context from research papers:\n{context}\n\n"
            f"Task: {task}\n\n"
            f"Provide a focused, expert response:"
        )
        answer = self._llm(prompt)
        return {"agent": self.name, "task": task,
                "answer": answer, "sources": sources}


class MultiAgentOrchestrator:
    """Hierarchical multi-agent system for complex research tasks.
    
    Architecture:
      Orchestrator — decomposes the question, assigns tasks to specialists,
                     synthesizes their findings into a final report
          ↓
      Retrieval Agent   — finds and summarizes relevant papers
      Analysis Agent    — identifies patterns, comparisons, tradeoffs  
      Critique Agent    — validates claims, flags unsupported statements
      Synthesis Agent   — combines findings into coherent narrative
    
    Why multi-agent over single agent:
    - Each agent has a focused context and role — better specialization
    - Agents can run in parallel (independent tasks)
    - Critique agent catches errors the generator would miss
    - Separates concerns: finding vs analyzing vs validating vs writing
    """

    def __init__(self, hybrid: HybridRetriever,
                 multi_query: MultiQueryRetriever,
                 reranker: BGEReranker,
                 generator: OllamaGenerator,
                 ollama_model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434"):
        self.model    = ollama_model
        self.base_url = base_url

        self.agents = {
            "retrieval": ResearchAgent(
                "Retrieval Agent",
                "research librarian who finds and summarizes relevant information",
                hybrid, reranker, ollama_model, base_url,
            ),
            "analysis": ResearchAgent(
                "Analysis Agent",
                "technical analyst who identifies patterns, tradeoffs, and comparisons",
                multi_query, reranker, ollama_model, base_url,
            ),
            "critique": ResearchAgent(
                "Critique Agent",
                "critical reviewer who validates claims and identifies gaps",
                hybrid, reranker, ollama_model, base_url,
            ),
        }
        self.generator = generator

    def _llm(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        return resp.json()["response"].strip()

    def _decompose(self, question: str) -> dict:
        prompt = (
            f"Decompose this research question into 3 specialized tasks.\n\n"
            f"Question: {question}\n\n"
            f"Output exactly this format:\n"
            f"RETRIEVAL: <what information to find>\n"
            f"ANALYSIS: <what patterns or tradeoffs to analyze>\n"
            f"CRITIQUE: <what claims or assumptions to validate>"
        )
        response = self._llm(prompt)
        tasks = {}
        for line in response.split("\n"):
            for key in ("RETRIEVAL", "ANALYSIS", "CRITIQUE"):
                if line.startswith(f"{key}:"):
                    tasks[key.lower()] = line.replace(f"{key}:", "").strip()
        if not tasks:
            tasks = {
                "retrieval": question,
                "analysis":  question,
                "critique":  question,
            }
        return tasks

    def _synthesize(self, question: str, findings: list[dict]) -> str:
        combined = ""
        for f in findings:
            combined += f"\n[{f['agent']}]\n{f['answer'][:600]}\n"

        prompt = (
            f"You are a senior research synthesizer. Combine these specialist "
            f"findings into a comprehensive, well-structured final answer.\n\n"
            f"Original Question: {question}\n\n"
            f"Specialist Findings:{combined}\n\n"
            f"Final Synthesized Answer:"
        )
        return self._llm(prompt)

    def run(self, question: str) -> dict:
        print(f"  Decomposing question...")
        tasks = self._decompose(question)

        findings = []
        for agent_name, task in tasks.items():
            print(f"  Running {agent_name} agent...")
            result = self.agents[agent_name].execute(task)
            findings.append(result)

        print(f"  Synthesizing findings...")
        final_answer = self._synthesize(question, findings)

        all_sources = []
        for f in findings:
            all_sources.extend(f["sources"])
        unique_sources = list(dict.fromkeys(all_sources))

        return {
            "question":    question,
            "answer":      final_answer,
            "tasks":       tasks,
            "findings":    findings,
            "sources":     unique_sources,
            "num_agents":  len(findings),
        }