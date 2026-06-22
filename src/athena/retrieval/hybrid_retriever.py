from collections import defaultdict
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.core import RetrievalResult

class HybridRetriever:
    """Combines dense + BM25 using Reciprocal Rank Fusion (k=60).
    
    RRF score = 1/(k + rank) summed across retrievers.
    Consistently improves recall by 5-15% over either alone.
    """

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k: int = 60):
        self.dense = dense
        self.bm25 = bm25
        self.k = k

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        dense_results = self.dense.retrieve(query, top_k=top_k * 3)
        bm25_results  = self.bm25.retrieve(query, top_k=top_k * 3)

        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(dense_results):
            cid = result.chunk.chunk_id
            rrf_scores[cid] += 1.0 / (self.k + rank + 1)
            chunk_map[cid] = result

        for rank, result in enumerate(bm25_results):
            cid = result.chunk.chunk_id
            rrf_scores[cid] += 1.0 / (self.k + rank + 1)
            chunk_map[cid] = result

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=chunk_map[cid].chunk, score=rrf_scores[cid])
            for cid in sorted_ids
        ]