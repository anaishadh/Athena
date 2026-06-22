import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator
from athena.pipelines.rag_pipeline import RAGPipeline

embedder  = BGEEmbedder()
store     = QdrantStore()
dense     = DenseRetriever(embedder, store)
bm25      = BM25Retriever()
bm25.load()
reranker  = BGEReranker()
generator = OllamaGenerator()
pipeline  = RAGPipeline(dense, bm25, reranker, generator)

question = "What are the main evaluation metrics used for RAG systems?"

print(f"Question: {question}\n")
result = pipeline.query(question)

print(f"Answer:\n{result['answer']}\n")
print(f"Sources:")
for s in set(result['sources']):
    print(f"  - {s[:70]}")