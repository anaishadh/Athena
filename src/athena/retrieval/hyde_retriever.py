import httpx
from athena.core import RetrievalResult
from athena.retrieval.qdrant_store import QdrantStore
from athena.ingestion.embedders.bge import BGEEmbedder

class HyDERetriever:
    """Hypothetical Document Embeddings.
    
    Instead of embedding the query directly, generates a hypothetical answer
    to the query using an LLM and embeds that instead. Works because the
    hypothetical answer is in the same form as corpus documents, placing it
    closer in embedding space to relevant chunks than a question would be.
    
    Tradeoff: adds one LLM call per query but improves recall on technical
    queries where question and answer have very different vocabulary.
    """

    def __init__(self, embedder: BGEEmbedder, store: QdrantStore,
                 ollama_model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434"):
        self.embedder = embedder
        self.store    = store
        self.model    = ollama_model
        self.base_url = base_url

    def _generate_hypothesis(self, query: str) -> str:
        prompt = (
            f"Write a concise 3-4 sentence factual paragraph that directly answers "
            f"this question as if it were a section in a research paper. "
            f"Be specific and technical.\n\nQuestion: {query}\n\nAnswer:"
        )
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        return resp.json()["response"].strip()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        hypothesis = self._generate_hypothesis(query)
        embedding  = self.embedder.embed([hypothesis])[0]
        return self.store.search(embedding, top_k=top_k)