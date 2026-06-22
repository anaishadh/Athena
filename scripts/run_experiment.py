import sys
import json
from pathlib import Path

sys.path.insert(0, "src")

import mlflow
from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator
from athena.pipelines.rag_pipeline import RAGPipeline
sys.path.insert(0, "scripts")
from run_evaluation import run_evaluation

mlflow.set_experiment("athena-rag")

embedder  = BGEEmbedder()
store     = QdrantStore()
dense     = DenseRetriever(embedder, store)
bm25      = BM25Retriever()
bm25.load()
reranker  = BGEReranker()
generator = OllamaGenerator()

experiments = [
    {
        "name":       "dense_only",
        "pipeline":   RAGPipeline(dense, bm25, reranker, generator,
                                  retrieve_k=20, rerank_k=5),
        "use_hybrid": False,
    },
    {
        "name":       "hybrid_reranked",
        "pipeline":   RAGPipeline(dense, bm25, reranker, generator,
                                  retrieve_k=20, rerank_k=5),
        "use_hybrid": True,
    },
]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):
        mlflow.log_params({
            "retriever":  "hybrid" if exp["use_hybrid"] else "dense",
            "embedder":   "BAAI/bge-m3",
            "reranker":   "BAAI/bge-reranker-v2-m3",
            "llm":        "qwen2.5:14b",
            "retrieve_k": 20,
            "rerank_k":   5,
        })

        results_path = f"data/golden/results_{exp['name']}.json"
        summary = run_evaluation(
            pipeline=exp["pipeline"],
            questions_path="data/golden/questions.json",
            results_path=results_path,
            run_name=exp["name"],
        )

        mlflow.log_metrics({
            "faithfulness": summary["avg_faithfulness"],
            "relevancy":    summary["avg_relevancy"],
            "correctness":  summary["avg_correctness"],
        })