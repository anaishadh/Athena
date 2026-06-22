from sentence_transformers import CrossEncoder
from athena.core import RetrievalResult

class BGEReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder via sentence-transformers.
    
    Sees query + document together — much more accurate than bi-encoder
    scoring. Use after retrieval: retrieve top-20, rerank, return top-5.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name, trust_remote_code=True)

    def rerank(self, query: str, results: list[RetrievalResult],
               top_k: int = 5) -> list[RetrievalResult]:
        if not results:
            return []
        pairs = [[query, r.chunk.text] for r in results]
        scores = self.model.predict(pairs)
        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(chunk=r.chunk, score=float(s))
            for s, r in scored[:top_k]
        ]