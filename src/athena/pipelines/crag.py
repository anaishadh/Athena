import httpx
import json
from athena.core import RetrievalResult
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator

class CRAGPipeline:
    """Corrective RAG — evaluates retrieved chunks before generation.
    
    Standard RAG blindly passes retrieved context to the LLM even when
    chunks are irrelevant. CRAG adds a quality gate:
    
    1. Retrieve top-k candidates
    2. Judge each as CORRECT / AMBIGUOUS / INCORRECT
    3. If enough correct chunks exist → generate normally
    4. If retrieval quality is low → fall back to broader BM25 search
    5. Generate only from quality-filtered context
    
    This reduces hallucination from irrelevant context and forces the
    system to surface retrieval failures explicitly rather than silently
    generating wrong answers.
    """

    def __init__(self, retriever: HybridRetriever,
                 reranker: BGEReranker,
                 generator: OllamaGenerator,
                 ollama_model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434",
                 retrieve_k: int = 10,
                 min_correct: int = 2):
        self.retriever   = retriever
        self.reranker    = reranker
        self.generator   = generator
        self.model       = ollama_model
        self.base_url    = base_url
        self.retrieve_k  = retrieve_k
        self.min_correct = min_correct

    def _evaluate_chunk(self, query: str, chunk_text: str) -> str:
        """Judge relevance of a single chunk. Returns CORRECT/AMBIGUOUS/INCORRECT."""
        prompt = (
            f"Query: {query}\n\n"
            f"Document excerpt:\n{chunk_text[:600]}\n\n"
            f"Is this document excerpt relevant to answering the query?\n"
            f"Reply with exactly one word: CORRECT, AMBIGUOUS, or INCORRECT."
        )
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        verdict = resp.json()["response"].strip().upper()
        for v in ("CORRECT", "AMBIGUOUS", "INCORRECT"):
            if v in verdict:
                return v
        return "AMBIGUOUS"

    def query(self, question: str) -> dict:
        # Step 1: retrieve candidates
        candidates = self.retriever.retrieve(question, top_k=self.retrieve_k)

        # Step 2: evaluate each chunk
        verdicts = []
        for result in candidates:
            verdict = self._evaluate_chunk(question, result.chunk.text)
            verdicts.append((result, verdict))

        # Step 3: filter by quality
        correct   = [r for r, v in verdicts if v == "CORRECT"]
        ambiguous = [r for r, v in verdicts if v == "AMBIGUOUS"]
        incorrect = [r for r, v in verdicts if v == "INCORRECT"]

        # Step 4: corrective action if not enough correct chunks
        if len(correct) >= self.min_correct:
            filtered = correct
            action   = "none"
        elif correct or ambiguous:
            filtered = correct + ambiguous
            action   = "partial_filter"
        else:
            # All chunks irrelevant — fall back to BM25 broader search
            filtered = self.retriever.bm25.retrieve(question, top_k=self.retrieve_k)
            action   = "bm25_fallback"

        # Step 5: rerank filtered chunks
        reranked = self.reranker.rerank(question, filtered, top_k=5)

        # Step 6: generate
        result = self.generator.generate(question, reranked)

        return {
            "question":        question,
            "answer":          result["answer"],
            "sources":         result["sources"],
            "chunks":          result["chunks"],
            "corrective_action": action,
            "verdict_summary": {
                "correct":   len(correct),
                "ambiguous": len(ambiguous),
                "incorrect": len(incorrect),
            },
        }