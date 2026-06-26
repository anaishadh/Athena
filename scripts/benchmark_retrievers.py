import sys
import json
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import mlflow
from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.retrieval.hyde_retriever import HyDERetriever
from athena.retrieval.multi_query_retriever import MultiQueryRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator
from athena.pipelines.rag_pipeline import RAGPipeline
from run_evaluation import run_evaluation

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("athena-retriever-benchmark")

embedder  = BGEEmbedder()
store     = QdrantStore()
dense     = DenseRetriever(embedder, store)
bm25      = BM25Retriever()
bm25.load()
reranker  = BGEReranker()
generator = OllamaGenerator()
hyde      = HyDERetriever(embedder, store)
multi     = MultiQueryRetriever(dense)

RETRIEVERS = {
    "dense":        dense,
    "bm25":         bm25,
    "hybrid":       HybridRetriever(dense, bm25),
    "hyde":         hyde,
    "multi_query":  multi,
}

for name, retriever in RETRIEVERS.items():
    print(f"\n{'='*50}")
    print(f"Benchmarking retriever: {name}")

    pipeline = RAGPipeline(
        dense=dense,
        bm25=bm25,
        reranker=reranker,
        generator=generator,
    )
    pipeline.hybrid.dense  = retriever if name not in ("hybrid",) else dense
    pipeline.hybrid.bm25   = bm25

    # Override retrieve method to use the target retriever
    original_retrieve = pipeline.hybrid.retrieve
    target = retriever

    def make_retrieve(r):
        def retrieve(query, top_k=10):
            return r.retrieve(query, top_k=top_k)
        return retrieve

    pipeline.hybrid.retrieve = make_retrieve(retriever)

    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "retriever": name,
            "embedder":  "BAAI/bge-m3",
            "reranker":  "BAAI/bge-reranker-v2-m3",
            "llm":       "qwen2.5:14b",
        })

        summary = run_evaluation(
            pipeline=pipeline,
            questions_path="data/golden/questions.json",
            results_path=f"data/golden/results_retriever_{name}.json",
            run_name=name,
        )

        mlflow.log_metrics({
            "faithfulness": summary["avg_faithfulness"],
            "relevancy":    summary["avg_relevancy"],
            "correctness":  summary["avg_correctness"],
        })

print("\nRetriever benchmark complete.")