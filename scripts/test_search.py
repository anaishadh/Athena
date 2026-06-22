import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore

embedder = BGEEmbedder()
store = QdrantStore()

query = "How does retrieval augmented generation work?"
embedding = embedder.embed_query(query)
results = store.search(embedding, top_k=3)

print(f"Query: {query}\n")
for i, r in enumerate(results, 1):
    print(f"Result {i} (score: {r.score:.3f})")
    print(f"  Title: {r.chunk.metadata.get('title', 'N/A')[:70]}")
    print(f"  Text:  {r.chunk.text[:150]}")
    print()