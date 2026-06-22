import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.reranking.bge_reranker import BGEReranker

embedder = BGEEmbedder()
store    = QdrantStore()
dense    = DenseRetriever(embedder, store)

bm25 = BM25Retriever()
bm25.load()

hybrid   = HybridRetriever(dense, bm25)
reranker = BGEReranker()

query = "What evaluation metrics are used for RAG systems?"

print(f"Query: {query}\n")

candidates = hybrid.retrieve(query, top_k=20)
reranked   = reranker.rerank(query, candidates, top_k=5)

print("Before reranking (Hybrid top-5):")
for i, r in enumerate(candidates[:5], 1):
    print(f"  {i}. (score: {r.score:.4f}) {r.chunk.metadata.get('title','')[:60]}")

print("\nAfter reranking (top-5):")
for i, r in enumerate(reranked, 1):
    print(f"  {i}. (score: {r.score:.4f}) {r.chunk.metadata.get('title','')[:60]}")