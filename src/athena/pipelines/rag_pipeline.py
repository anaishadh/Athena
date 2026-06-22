from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator

class RAGPipeline:
    """Full RAG pipeline: Hybrid Retrieval → Reranking → Generation.
    
    retrieve_k: number of candidates fetched before reranking
    rerank_k:   number of chunks passed to the LLM after reranking
    """

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever,
                 reranker: BGEReranker, generator: OllamaGenerator,
                 retrieve_k: int = 20, rerank_k: int = 5):
        self.hybrid    = HybridRetriever(dense, bm25)
        self.reranker  = reranker
        self.generator = generator
        self.retrieve_k = retrieve_k
        self.rerank_k   = rerank_k

    def query(self, question: str) -> dict:
        candidates = self.hybrid.retrieve(question, top_k=self.retrieve_k)
        reranked   = self.reranker.rerank(question, candidates, top_k=self.rerank_k)
        result     = self.generator.generate(question, reranked)

        return {
            "question":    question,
            "answer":      result["answer"],
            "sources":     result["sources"],
            "chunks":      result["chunks"],
            "num_retrieved": len(candidates),
            "num_reranked":  len(reranked),
        }