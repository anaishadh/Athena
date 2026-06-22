import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.core import RetrievalResult

class DenseRetriever:
    def __init__(self, embedder: BGEEmbedder, store: QdrantStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        embedding = self.embedder.embed_query(query)
        return self.store.search(embedding, top_k=top_k)