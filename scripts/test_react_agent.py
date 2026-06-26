import sys
sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.agents.react_agent import ReActAgent

embedder = BGEEmbedder()
store    = QdrantStore()
dense    = DenseRetriever(embedder, store)
bm25     = BM25Retriever()
bm25.load()
hybrid   = HybridRetriever(dense, bm25)
reranker = BGEReranker()

agent = ReActAgent(hybrid, reranker)

questions = [
    "What are the key differences between ColBERT and standard dense retrieval, and when would you choose one over the other?",
    "How do RAGAS metrics measure RAG system quality and what are their limitations?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Question: {q}")
    print(f"{'='*60}")
    result = agent.run(q)
    print(f"\nIterations: {result['iterations']} | Searches: {result['searches']}")
    print(f"\nReasoning trace:")
    for i, (thought, action) in enumerate(zip(result['thoughts'], result['tool_calls']), 1):
        print(f"  [{i}] Thought: {thought[:100]}")
        print(f"       Action:  {action}")
    print(f"\nFinal Answer:\n{result['answer'][:400]}")