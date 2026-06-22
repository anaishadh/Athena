# Athena — Enterprise AI Research Intelligence Platform

A production-grade RAG (Retrieval-Augmented Generation) system built to demonstrate systematic AI engineering competence. Athena ingests 100 AI/ML research papers and answers natural language queries with grounded, cited responses.

## Architecture
Query → Hybrid Retrieval (Dense + BM25) → Reranking → Generation → Evaluated Answer

**Retrieval**: BGE-M3 embeddings (1024-dim) stored in Qdrant, combined with BM25 sparse retrieval via Reciprocal Rank Fusion (k=60)

**Reranking**: BAAI/bge-reranker-v2-m3 cross-encoder re-scores top-20 candidates, returns top-5

**Generation**: Qwen2.5-14B via Ollama generates grounded answers from reranked context

**Evaluation**: LLM-as-judge scoring faithfulness, relevancy, and correctness across 20 golden questions, tracked in MLflow

## Benchmark Results

| Pipeline | Faithfulness | Relevancy | Correctness |
|---|---|---|---|
| Dense only | 0.52 | 0.73 | 0.55 |
| Hybrid + Reranked | 0.59 | 0.71 | 0.62 |

Hybrid retrieval with reranking improves correctness by +12.7% over dense-only retrieval.

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | BAAI/bge-m3 (open source, 1024-dim) |
| Vector DB | Qdrant |
| Sparse retrieval | BM25 (rank-bm25) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | Qwen2.5-14B via Ollama |
| Experiment tracking | MLflow |
| Package manager | uv |

## Project Structure

    src/athena/
    ├── ingestion/
    │   ├── loaders/        # PDF loading via PyMuPDF
    │   ├── chunkers/       # Fixed, sliding, recursive, metadata-aware
    │   └── embedders/      # BGE-M3 open-source embedder
    ├── retrieval/
    │   ├── dense_retriever.py    # Semantic search via Qdrant
    │   ├── bm25_retriever.py     # Exact-term sparse retrieval
    │   └── hybrid_retriever.py   # RRF fusion of dense + sparse
    ├── reranking/
    │   └── bge_reranker.py       # Cross-encoder reranking
    └── pipelines/
        ├── generator.py          # Qwen2.5-14B answer generation
        └── rag_pipeline.py       # End-to-end pipeline

## Setup

**Requirements**: Python 3.11, Docker, Ollama, CUDA GPU recommended

```bash
# Install dependencies
uv install

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Pull LLM
ollama pull qwen2.5:14b

# Fetch and ingest corpus
uv run python scripts/fetch_corpus.py
uv run python scripts/ingest_corpus.py
uv run python scripts/build_bm25_index.py

# Run evaluation
uv run python scripts/run_evaluation.py
```

## Corpus

100 arXiv papers spanning:
- Retrieval-Augmented Generation
- Embedding models and contrastive learning
- Transformer architectures and LLMs
- LLM agent systems and reasoning
- Vector database and ANN algorithms

## Evaluation

20 manually curated golden questions with reference answers. Each pipeline run is scored by an LLM judge (Qwen2.5-14B) on three dimensions and logged to MLflow for comparison.

```bash
uv run mlflow ui --port 5000
```