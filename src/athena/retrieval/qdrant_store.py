from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from athena.core import Chunk, RetrievalResult
import uuid

class QdrantStore:
    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection_name: str = "athena_papers", dimension: int = 1024):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]):
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id or str(i))),
                vector=embeddings[i],
                payload={"text": c.text, "metadata": c.metadata, "chunk_id": c.chunk_id},
            )
            for i, c in enumerate(chunks)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[RetrievalResult]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
        ).points
        return [
            RetrievalResult(
                chunk=Chunk(
                    text=r.payload["text"],
                    metadata=r.payload["metadata"],
                    chunk_id=r.payload["chunk_id"],
                ),
                score=r.score,
            )
            for r in results
        ]