import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator
from athena.pipelines.crag import CRAGPipeline

embedder  = BGEEmbedder()
store     = QdrantStore()
dense     = DenseRetriever(embedder, store)
bm25      = BM25Retriever()
bm25.load()
hybrid    = HybridRetriever(dense, bm25)
reranker  = BGEReranker()
generator = OllamaGenerator()

pipeline = CRAGPipeline(hybrid, reranker, generator)

questions = [
    "What evaluation metrics are used for RAG systems?",
    "What is the capital of Mars?",  # should trigger fallback
    "How does HNSW enable fast vector search?",
]

for q in questions:
    print(f"\nQuestion: {q}")
    result = pipeline.query(q)
    print(f"Action:   {result['corrective_action']}")
    print(f"Verdicts: {result['verdict_summary']}")
    print(f"Answer:   {result['answer'][:200]}")
    print("-" * 60)