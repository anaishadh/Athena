import json
import hashlib
import redis
import numpy as np
from athena.ingestion.embedders.bge import BGEEmbedder


class SemanticCache:
    """Semantic cache using Redis — avoids re-running retrieval and generation
    for semantically similar queries.

    Unlike exact-match caching (same string = cache hit), semantic caching
    embeds the query and finds cached responses whose query embeddings are
    close in vector space.

    Example:
      Query 1: "What is RAG?"          → cache miss → run pipeline → cache result
      Query 2: "Explain RAG to me"     → cache hit (cosine sim > threshold)
      Query 3: "How does RAG work?"    → cache hit (cosine sim > threshold)

    This saves 20-30 seconds per cache hit since we skip Qwen generation entirely.
    Cache hit rate of 20-30% is typical in production — significant cost savings.
    """

    def __init__(self, embedder: BGEEmbedder,
                 host: str = "localhost", port: int = 6379,
                 threshold: float = 0.92, ttl: int = 3600):
        self.embedder  = embedder
        self.client    = redis.Redis(host=host, port=port,
                                     decode_responses=True)
        self.threshold = threshold
        self.ttl       = ttl
        self.key_prefix = "athena:cache:"

    def _embedding_key(self, query: str) -> str:
        return self.key_prefix + hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> dict | None:
        """Check if a semantically similar query is cached."""
        query_embedding = np.array(self.embedder.embed_query(query))

        # Scan all cached embeddings
        keys = list(self.client.scan_iter(f"{self.key_prefix}*"))
        for key in keys:
            cached = self.client.get(key)
            if not cached:
                continue
            try:
                data = json.loads(cached)
                cached_embedding = np.array(data["embedding"])
                similarity = float(np.dot(query_embedding, cached_embedding))
                if similarity >= self.threshold:
                    return {
                        "answer":     data["answer"],
                        "sources":    data["sources"],
                        "cache_hit":  True,
                        "similarity": round(similarity, 4),
                    }
            except Exception:
                continue
        return None

    def set(self, query: str, answer: str, sources: list[str]):
        """Cache the query embedding and response."""
        embedding = self.embedder.embed_query(query)
        data = {
            "query":     query,
            "answer":    answer,
            "sources":   sources,
            "embedding": embedding,
        }
        key = self._embedding_key(query)
        self.client.setex(key, self.ttl, json.dumps(data))

    def clear(self):
        keys = list(self.client.scan_iter(f"{self.key_prefix}*"))
        if keys:
            self.client.delete(*keys)
        return len(keys)