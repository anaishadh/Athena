# Athena — Enterprise AI Research Intelligence Platform

A production-grade RAG (Retrieval-Augmented Generation) system built to demonstrate systematic AI engineering competence. Athena ingests 100 AI/ML research papers and answers natural language queries with grounded, cited responses.

## Architecture
Query → Hybrid Retrieval (Dense + BM25) → Reranking → Generation → Evaluated Answer

**Retrieval**: BGE-M3 embeddings (1024-dim) stored in Qdrant, combined with BM25 sparse retrieval via Reciprocal Rank Fusion (k=60)

**Reranking**: BAAI/bge-reranker-v2-m3 cross-encoder re-scores top-20 candidates, returns top-5

**Generation**: Qwen2.5-14B via Ollama generates grounded answers from reranked context

**Evaluation**: LLM-as-judge scoring faithfulness, relevancy, and correctness across 20 golden questions, tracked in MLflow

## Benchmark Results

### Chunking Strategy Comparison
| Chunker | Faithfulness | Relevancy | Correctness | Chunks | Ingest Time |
|---|---|---|---|---|---|
| fixed | 0.63 | 0.86 | 0.70 | 2795 | 42 min |
| sliding | 0.54 | 0.83 | 0.66 | 3377 | 48 min |
| recursive | 0.55 | 0.77 | 0.61 | 2805 | 42 min |
| metadata-aware | 0.48 | 0.72 | 0.55 | 2805 | 71 min |
| parent-child | 0.64 | 0.76 | 0.69 | 4845 | 70 min |
| semantic | 0.65 | 0.80 | 0.69 | 10670 | 101 min |

### Retrieval Strategy Comparison
| Retriever | Faithfulness | Relevancy | Correctness |
|---|---|---|---|
| dense | 0.40 | 0.66 | 0.51 |
| bm25 | 0.53 | 0.83 | 0.66 |
| hybrid (RRF) | 0.62 | 0.78 | 0.66 |
| hyde | 0.63 | 0.79 | 0.64 |
| multi-query | 0.60 | **0.87** | 0.68 |

All experiments tracked in MLflow. Multi-query retrieval achieves highest relevancy (0.87). BM25 outperforms dense on this terminology-heavy corpus, confirming that exact keyword matching is critical for technical document retrieval.

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

```
src/athena/
├── core.py                          # Shared data models (Document, Chunk, RetrievalResult)
├── ingestion/
│   ├── loaders/
│   │   └── pdf_loader.py            # PDF text extraction via PyMuPDF
│   ├── chunkers/
│   │   ├── fixed.py                 # Fixed-size word count chunking
│   │   ├── sliding.py               # Sliding window with overlap
│   │   ├── recursive.py             # Recursive paragraph → sentence splitting
│   │   ├── semantic.py              # Embedding similarity breakpoint chunking
│   │   ├── parent_child.py          # Small child chunks, large parent context
│   │   └── metadata_aware.py        # Wraps any chunker, prepends paper metadata
│   └── embedders/
│       └── bge.py                   # BAAI/bge-m3 local GPU embedder
├── retrieval/
│   ├── qdrant_store.py              # Qdrant vector store (HNSW, cosine)
│   ├── dense_retriever.py           # Semantic search via BGE-M3 + Qdrant
│   ├── bm25_retriever.py            # Exact-term sparse retrieval
│   ├── hybrid_retriever.py          # Dense + BM25 fusion via RRF (k=60)
│   ├── hyde_retriever.py            # Hypothetical document embeddings
│   └── multi_query_retriever.py     # N alternative phrasings via RRF
├── reranking/
│   └── bge_reranker.py              # BAAI/bge-reranker-v2-m3 cross-encoder
├── pipelines/
│   ├── generator.py                 # Qwen2.5-14B generation via Ollama
│   └── rag_pipeline.py              # End-to-end hybrid → rerank → generate
├── evaluation/                      # (in progress)
├── agents/                          # (in progress)
└── api/                             # (in progress)
```

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

20 manually curated golden questions with reference answers. Each pipeline variant is scored by an LLM judge (Qwen2.5-14B) on three dimensions:

- **Faithfulness** — are all claims supported by retrieved context?
- **Relevancy** — does the answer address the question?
- **Correctness** — does the answer align with the ground truth?

All runs logged to MLflow for side-by-side comparison.

```bash
uv run mlflow ui --port 5000
```