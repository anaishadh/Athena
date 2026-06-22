import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever

embedder = BGEEmbedder()
store    = QdrantStore()
dense    = DenseRetriever(embedder, store)

bm25 = BM25Retriever()
bm25.load()

hybrid = HybridRetriever(dense, bm25)

queries = [
    "How does attention mechanism work in transformers?",
    "HNSW approximate nearest neighbor search",
    "evaluation metrics for RAG systems",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    for name, retriever in [("Dense", dense), ("BM25", bm25), ("Hybrid", hybrid)]:
        results = retriever.retrieve(query, top_k=3)
        print(f"\n  [{name}]")
        for i, r in enumerate(results, 1):
            title = r.chunk.metadata.get("title", "N/A")[:55]
            print(f"  {i}. (score: {r.score:.4f}) {title}")