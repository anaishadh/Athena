import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator
from athena.agents.planner_executor import PlannerExecutorAgent

embedder  = BGEEmbedder()
store     = QdrantStore()
dense     = DenseRetriever(embedder, store)
bm25      = BM25Retriever()
bm25.load()
hybrid    = HybridRetriever(dense, bm25)
reranker  = BGEReranker()
generator = OllamaGenerator()

agent = PlannerExecutorAgent(hybrid, reranker, generator)

question = "Compare the tradeoffs between dense retrieval, sparse retrieval, and hybrid retrieval approaches in terms of accuracy, speed, and use cases."

print(f"Question: {question}\n")
result = agent.run(question)

print(f"\nPlan ({result['num_steps']} steps):")
for i, step in enumerate(result['plan'], 1):
    print(f"  {i}. {step}")

print(f"\nStep Results:")
for i, r in enumerate(result['step_results'], 1):
    print(f"  [{i}] {r['step'][:60]}")
    print(f"       {r['answer'][:150]}")

print(f"\nFinal Answer:\n{result['answer'][:500]}")