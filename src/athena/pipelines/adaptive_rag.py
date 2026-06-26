import httpx
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.retrieval.hyde_retriever import HyDERetriever
from athena.retrieval.multi_query_retriever import MultiQueryRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator

class AdaptiveRAGPipeline:
    """Routes queries to different retrieval pipelines based on complexity.
    
    Simple factual queries don't need expensive multi-query retrieval.
    Complex multi-aspect queries shouldn't use cheap single-pass retrieval.
    Adaptive RAG classifies first, then routes to the optimal pipeline.
    
    L1 — Simple factual: direct LLM answer, no retrieval needed
    L2 — Single concept: dense retrieval + rerank + generate
    L3 — Technical/specific: hybrid retrieval + rerank + generate
    L4 — Complex/multi-hop: multi-query retrieval + rerank + generate
    """

    def __init__(self,
                 dense: DenseRetriever,
                 bm25: BM25Retriever,
                 hyde: HyDERetriever,
                 multi_query: MultiQueryRetriever,
                 reranker: BGEReranker,
                 generator: OllamaGenerator,
                 ollama_model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434"):
        self.dense       = dense
        self.bm25        = bm25
        self.hybrid      = HybridRetriever(dense, bm25)
        self.hyde        = hyde
        self.multi_query = multi_query
        self.reranker    = reranker
        self.generator   = generator
        self.model       = ollama_model
        self.base_url    = base_url

    def _classify(self, query: str) -> str:
        prompt = (
            f"Classify this search query into exactly one complexity level:\n\n"
            f"L1 = Simple fact, answerable from general knowledge (e.g. 'What is an embedding?')\n"
            f"L2 = Single concept needing one document (e.g. 'How does BM25 score documents?')\n"
            f"L3 = Technical query needing precise terminology match (e.g. 'HNSW graph construction algorithm')\n"
            f"L4 = Complex query needing multiple sources (e.g. 'Compare dense vs sparse retrieval tradeoffs')\n\n"
            f"Query: {query}\n\n"
            f"Reply with only: L1, L2, L3, or L4."
        )
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        text = resp.json()["response"].strip().upper()
        for level in ("L4", "L3", "L2", "L1"):
            if level in text:
                return level
        return "L2"

    def _direct_answer(self, query: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model,
                  "prompt": f"Answer concisely: {query}",
                  "stream": False},
            timeout=60,
        )
        return resp.json()["response"].strip()

    def query(self, question: str) -> dict:
        level = self._classify(question)

        if level == "L1":
            answer = self._direct_answer(question)
            return {
                "question":      question,
                "answer":        answer,
                "routing_level": level,
                "retriever":     "none",
                "sources":       [],
                "chunks":        [],
            }

        # Select retriever based on level
        if level == "L2":
            retriever = self.dense
        elif level == "L3":
            retriever = self.hybrid
        else:  # L4
            retriever = self.multi_query

        candidates = retriever.retrieve(question, top_k=20)
        reranked   = self.reranker.rerank(question, candidates, top_k=5)
        result     = self.generator.generate(question, reranked)

        return {
            "question":      question,
            "answer":        result["answer"],
            "routing_level": level,
            "retriever":     level,
            "sources":       result["sources"],
            "chunks":        result["chunks"],
        }