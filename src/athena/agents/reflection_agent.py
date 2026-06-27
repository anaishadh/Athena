import httpx
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.reranking.bge_reranker import BGEReranker


class ReflectionAgent:
    """Reflection / Self-Critique agent — iteratively improves its own answers.
    
    Process:
    1. GENERATE: produce an initial answer from retrieved context
    2. CRITIQUE: critically evaluate the answer for gaps and errors
    3. REVISE: generate an improved answer incorporating the critique
    4. Repeat until quality threshold met or max iterations reached
    
    Why this works: the same LLM that generated the answer evaluates it
    with fresh context (no generation bias). Self-critique consistently
    surfaces missing aspects, unsupported claims, and unclear explanations.
    
    Key design decision: cap iterations at 3. Without a cap the agent
    over-refines and can degrade quality by finding nonexistent problems.
    Also: if critique says 'no issues found', stop immediately.
    """

    def __init__(self, retriever: HybridRetriever,
                 reranker: BGEReranker,
                 ollama_model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434",
                 max_iterations: int = 3):
        self.retriever  = retriever
        self.reranker   = reranker
        self.model      = ollama_model
        self.base_url   = base_url
        self.max_iter   = max_iterations

    def _llm(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        return resp.json()["response"].strip()

    def _retrieve_context(self, query: str) -> str:
        candidates = self.retriever.retrieve(query, top_k=10)
        reranked   = self.reranker.rerank(query, candidates, top_k=5)
        return "\n\n".join(r.chunk.text for r in reranked)

    def _generate(self, question: str, context: str) -> str:
        prompt = (
            f"You are a research assistant. Answer this question using "
            f"only the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        return self._llm(prompt)

    def _critique(self, question: str, answer: str, context: str) -> str:
        prompt = (
            f"You are a critical evaluator. Review this answer strictly.\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            f"Context used: {context[:1000]}\n\n"
            f"Identify specific problems:\n"
            f"1. Claims not supported by context (hallucinations)\n"
            f"2. Important aspects of the question not addressed\n"
            f"3. Unclear or imprecise explanations\n\n"
            f"If the answer is good and complete, respond with: NO_ISSUES\n"
            f"Otherwise list the specific problems:"
        )
        return self._llm(prompt)

    def _revise(self, question: str, answer: str,
                critique: str, context: str) -> str:
        prompt = (
            f"You are a research assistant. Improve this answer based on "
            f"the critique provided.\n\n"
            f"Original Question: {question}\n\n"
            f"Previous Answer: {answer}\n\n"
            f"Critique: {critique}\n\n"
            f"Context: {context}\n\n"
            f"Revised Answer (address all critique points):"
        )
        return self._llm(prompt)

    def run(self, question: str) -> dict:
        context  = self._retrieve_context(question)
        answer   = self._generate(question, context)
        history  = [{"iteration": 0, "answer": answer, "critique": None}]

        for i in range(1, self.max_iter + 1):
            critique = self._critique(question, answer, context)
            print(f"  Iteration {i} critique: {critique[:100]}")

            if "NO_ISSUES" in critique.upper():
                print(f"  ✓ No issues found — stopping at iteration {i}")
                history[-1]["critique"] = critique
                break

            answer = self._revise(question, answer, critique, context)
            history.append({
                "iteration": i,
                "answer":    answer,
                "critique":  critique,
            })

        return {
            "question":   question,
            "answer":     answer,
            "iterations": len(history),
            "history":    history,
        }