import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.agents.reflection_agent import ReflectionAgent

embedder = BGEEmbedder()
store    = QdrantStore()
dense    = DenseRetriever(embedder, store)
bm25     = BM25Retriever()
bm25.load()
hybrid   = HybridRetriever(dense, bm25)
reranker = BGEReranker()

agent = ReflectionAgent(hybrid, reranker, max_iterations=3)

question = "What are the limitations of using LLM-as-judge for RAG evaluation and how can they be mitigated?"

print(f"Question: {question}\n")
result = agent.run(question)

print(f"\nCompleted in {result['iterations']} iteration(s)")
print(f"\nFinal Answer:\n{result['answer'][:600]}")
print(f"\nIteration History:")
for h in result['history']:
    print(f"  Iteration {h['iteration']}:")
    print(f"    Answer:  {h['answer'][:100]}")
    if h['critique']:
        print(f"    Critique: {h['critique'][:100]}")