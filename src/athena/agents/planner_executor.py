import httpx
import json
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator


class PlannerExecutorAgent:
    """Planner-Executor pattern — separates planning from execution.
    
    Unlike ReAct which reasons step-by-step, Planner-Executor:
    1. PLANNER: generates a complete research plan upfront (one LLM call)
    2. EXECUTOR: executes each step independently (parallel-ready)
    3. SYNTHESIZER: combines all step results into a final answer
    
    Advantages over ReAct:
    - Plan is visible and interpretable before execution
    - Independent steps can run in parallel (reduces latency)
    - Easier to debug — examine the plan vs examine execution
    - Natural human-in-the-loop: approve plan before executing
    
    Best for: complex multi-aspect research questions where the
    sub-tasks are known upfront and largely independent.
    """

    def __init__(self, retriever: HybridRetriever,
                 reranker: BGEReranker,
                 generator: OllamaGenerator,
                 ollama_model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434",
                 max_steps: int = 4):
        self.retriever  = retriever
        self.reranker   = reranker
        self.generator  = generator
        self.model      = ollama_model
        self.base_url   = base_url
        self.max_steps  = max_steps

    def _llm(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        return resp.json()["response"].strip()

    def _plan(self, question: str) -> list[str]:
        prompt = (
            f"You are a research planner. Break this complex question into "
            f"{self.max_steps} independent search sub-tasks. Each sub-task "
            f"should be a specific search query that retrieves one aspect of "
            f"the answer. Output only a JSON array of strings.\n\n"
            f"Question: {question}"
        )
        response = self._llm(prompt)
        try:
            start = response.index("[")
            end   = response.rindex("]") + 1
            steps = json.loads(response[start:end])
            return steps[:self.max_steps]
        except Exception:
            return [question]

    def _execute_step(self, step: str) -> dict:
        candidates = self.retriever.retrieve(step, top_k=10)
        reranked   = self.reranker.rerank(step, candidates, top_k=3)
        context    = "\n\n".join(r.chunk.text for r in reranked)
        answer     = self._llm(
            f"Based on this context, answer concisely:\n{context}\n\nQuestion: {step}"
        )
        return {
            "step":    step,
            "answer":  answer,
            "sources": [r.chunk.metadata.get("title", "") for r in reranked],
        }

    def _synthesize(self, question: str, step_results: list[dict]) -> str:
        findings = ""
        for i, r in enumerate(step_results, 1):
            findings += f"\nFinding {i} ({r['step']}):\n{r['answer']}\n"

        prompt = (
            f"You are a research synthesizer. Combine these findings into "
            f"a comprehensive, well-structured answer to the original question.\n\n"
            f"Original Question: {question}\n\n"
            f"Findings:{findings}\n\n"
            f"Synthesized Answer:"
        )
        return self._llm(prompt)

    def run(self, question: str) -> dict:
        # Step 1: Plan
        plan = self._plan(question)
        print(f"  Plan: {plan}")

        # Step 2: Execute each step
        step_results = []
        for step in plan:
            result = self._execute_step(step)
            step_results.append(result)
            print(f"  ✓ Executed: {step[:60]}")

        # Step 3: Synthesize
        final_answer = self._synthesize(question, step_results)

        return {
            "question":     question,
            "plan":         plan,
            "step_results": step_results,
            "answer":       final_answer,
            "num_steps":    len(plan),
        }