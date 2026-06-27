import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.retrieval.multi_query_retriever import MultiQueryRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator
from athena.agents.multi_agent import MultiAgentOrchestrator

embedder    = BGEEmbedder()
store       = QdrantStore()
dense       = DenseRetriever(embedder, store)
bm25        = BM25Retriever()
bm25.load()
hybrid      = HybridRetriever(dense, bm25)
multi_query = MultiQueryRetriever(dense)
reranker    = BGEReranker()
generator   = OllamaGenerator()

orchestrator = MultiAgentOrchestrator(hybrid, multi_query, reranker, generator)

question = "What are the key architectural decisions when building a production RAG system, and what are the tradeoffs of each decision?"

print(f"Question: {question}\n")
result = orchestrator.run(question)

print(f"\nTask Decomposition:")
for agent, task in result['tasks'].items():
    print(f"  {agent}: {task[:80]}")

print(f"\nAgent Findings:")
for f in result['findings']:
    print(f"\n  [{f['agent']}]")
    print(f"  {f['answer'][:200]}")

print(f"\nFinal Answer:\n{result['answer'][:600]}")
print(f"\nSources ({len(result['sources'])}):")
for s in result['sources'][:5]:
    print(f"  - {s[:70]}")