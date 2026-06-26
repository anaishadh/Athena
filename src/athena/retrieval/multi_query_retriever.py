import httpx
import json
from collections import defaultdict
from athena.core import RetrievalResult
from athena.retrieval.dense_retriever import DenseRetriever

class MultiQueryRetriever:
    """Generates N alternative phrasings of the query, retrieves for each,
    merges results via RRF.
    
    Improves recall for ambiguous or complex queries because different phrasings
    surface different relevant documents. A query about 'attention mechanisms'
    might miss papers that describe the same concept as 'self-attention' or
    'scaled dot-product attention'.
    """

    def __init__(self, dense: DenseRetriever, n_queries: int = 4,
                 ollama_model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434",
                 k: int = 60):
        self.dense     = dense
        self.n_queries = n_queries
        self.model     = ollama_model
        self.base_url  = base_url
        self.k         = k

    def _generate_queries(self, query: str) -> list[str]:
        prompt = (
            f"Generate {self.n_queries} different search query phrasings for "
            f"finding research papers about the following topic. "
            f"Use different vocabulary and perspectives. "
            f"Output only a JSON array of strings, no explanation.\n\n"
            f"Query: {query}"
        )
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        text = resp.json()["response"].strip()
        try:
            start = text.index("[")
            end   = text.rindex("]") + 1
            return json.loads(text[start:end])[:self.n_queries]
        except Exception:
            return [query]

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        queries = [query] + self._generate_queries(query)

        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map:  dict[str, RetrievalResult] = {}

        for q in queries:
            results = self.dense.retrieve(q, top_k=top_k * 2)
            for rank, result in enumerate(results):
                cid = result.chunk.chunk_id
                rrf_scores[cid] += 1.0 / (self.k + rank + 1)
                chunk_map[cid]   = result

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=chunk_map[cid].chunk, score=rrf_scores[cid])
            for cid in sorted_ids
        ]