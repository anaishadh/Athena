import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hyde_retriever import HyDERetriever
from athena.retrieval.multi_query_retriever import MultiQueryRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator
from athena.pipelines.adaptive_rag import AdaptiveRAGPipeline

embedder    = BGEEmbedder()
store       = QdrantStore()
dense       = DenseRetriever(embedder, store)
bm25        = BM25Retriever()
bm25.load()
hyde        = HyDERetriever(embedder, store)
multi_query = MultiQueryRetriever(dense)
reranker    = BGEReranker()
generator   = OllamaGenerator()

pipeline = AdaptiveRAGPipeline(
    dense, bm25, hyde, multi_query, reranker, generator
)

questions = [
    "What is an embedding?",
    "How does BM25 score documents?",
    "HNSW graph construction algorithm complexity",
    "Compare dense vs sparse retrieval tradeoffs for technical corpora",
]

for q in questions:
    print(f"\nQuestion: {q}")
    result = pipeline.query(q)
    print(f"Routed to: {result['routing_level']} ({result['retriever']})")
    print(f"Answer:    {result['answer'][:200]}")
    print("-" * 60)