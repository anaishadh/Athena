import sys
import json
from pathlib import Path

sys.path.insert(0, "src")

import mlflow
import time
from athena.ingestion.loaders.pdf_loader import PDFLoader
from athena.ingestion.chunkers.fixed import FixedChunker
from athena.ingestion.chunkers.sliding import SlidingWindowChunker
from athena.ingestion.chunkers.recursive import RecursiveChunker
from athena.ingestion.chunkers.semantic import SemanticChunker
from athena.ingestion.chunkers.parent_child import ParentChildChunker
from athena.ingestion.chunkers.metadata_aware import MetadataAwareWrapper
from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator
from athena.pipelines.rag_pipeline import RAGPipeline

sys.path.insert(0, "scripts")
from run_evaluation import run_evaluation

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("athena-chunker-benchmark")

metadata  = json.loads(Path("data/corpus/metadata.json").read_text())
embedder  = BGEEmbedder()
reranker  = BGEReranker()
generator = OllamaGenerator()
loader    = PDFLoader()

CHUNKERS = {
    # "fixed":        FixedChunker(chunk_size=512, overlap=0),
    # "sliding":      SlidingWindowChunker(chunk_size=512, overlap=128),
    # "recursive":    RecursiveChunker(chunk_size=512, overlap=64),
    # "metadata":     MetadataAwareWrapper(RecursiveChunker(chunk_size=512, overlap=64)),
    # "parent_child": MetadataAwareWrapper(ParentChildChunker()),
    "semantic":     MetadataAwareWrapper(SemanticChunker(embedder)),
}

for chunker_name, chunker in CHUNKERS.items():
    print(f"\n{'='*50}")
    print(f"Benchmarking chunker: {chunker_name}")

    # Ingest with this chunker
    collection = f"athena_{chunker_name}"
    store = QdrantStore(collection_name=collection)

    # Delete if exists
    try:
        store.client.delete_collection(collection)
    except Exception:
        pass
    store._ensure_collection()

    all_chunks = []
    start_ingest = time.time()

    for paper in metadata:
        pdf_path = paper["pdf_path"]
        if not Path(pdf_path).exists():
            continue
        docs = loader.load(pdf_path)
        for doc in docs:
            doc.metadata.update({
                "title":      paper["title"],
                "authors":    paper["authors"],
                "published":  paper["published"],
                "categories": paper["categories"],
                "paper_id":   paper["id"],
            })
        try:
            chunks = chunker.chunk(docs)
        except Exception as e:
            print(f"  ⚠ Skipping paper {paper['id']}: {e}")
            continue
        if not chunks:
            continue
        texts      = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
        store.add_chunks(chunks, embeddings)
        all_chunks.extend(chunks)

    ingest_time = time.time() - start_ingest
    print(f"  Chunks: {len(all_chunks)} | Ingest time: {ingest_time:.1f}s")

    # Build BM25
    bm25 = BM25Retriever()
    bm25.index(all_chunks)

    # Evaluate
    dense    = DenseRetriever(embedder, store)
    pipeline = RAGPipeline(dense, bm25, reranker, generator)

    with mlflow.start_run(run_name=chunker_name):
        mlflow.log_params({
            "chunker":    chunker_name,
            "embedder":   "BAAI/bge-m3",
            "reranker":   "BAAI/bge-reranker-v2-m3",
            "llm":        "qwen2.5:14b",
            "num_chunks": len(all_chunks),
            "ingest_time_seconds": round(ingest_time, 1),
        })

        summary = run_evaluation(
            pipeline=pipeline,
            questions_path="data/golden/questions.json",
            results_path=f"data/golden/results_{chunker_name}.json",
            run_name=chunker_name,
        )

        mlflow.log_metrics({
            "faithfulness":       summary["avg_faithfulness"],
            "relevancy":          summary["avg_relevancy"],
            "correctness":        summary["avg_correctness"],
            "num_chunks":         len(all_chunks),
            "ingest_time_seconds": round(ingest_time, 1),
        })

print("\nBenchmark complete. View results: mlflow ui --port 5000")