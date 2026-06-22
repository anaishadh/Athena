import sys
sys.path.insert(0, "src")

from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.core import Chunk

print("Loading all chunks from Qdrant...")
store = QdrantStore()

all_chunks = []
limit = 100
offset = None

while True:
    results, next_offset = store.client.scroll(
        collection_name=store.collection_name,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    for r in results:
        all_chunks.append(Chunk(
            text=r.payload["text"],
            metadata=r.payload["metadata"],
            chunk_id=r.payload["chunk_id"],
        ))
    if next_offset is None:
        break
    offset = next_offset

print(f"Loaded {len(all_chunks)} chunks")

bm25 = BM25Retriever()
bm25.index(all_chunks)
bm25.save()

print("BM25 index saved to data/bm25_chunks.json")