import httpx
from athena.core import RetrievalResult

class OllamaGenerator:
    """Qwen2.5:14b via Ollama for answer generation.
    
    Takes retrieved chunks as context and generates a grounded answer.
    Instructs the model to only use provided context — reduces hallucination.
    """

    def __init__(self, model: str = "qwen2.5:14b",
                 base_url: str = "http://localhost:11434"):
        self.model    = model
        self.base_url = base_url

    def generate(self, query: str, results: list[RetrievalResult]) -> dict:
        context = "\n\n---\n\n".join(
            f"[Source: {r.chunk.metadata.get('title', 'Unknown')}]\n{r.chunk.text}"
            for r in results
        )

        prompt = f"""You are a research assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so explicitly.
Do not make up information.

Context:
{context}

Question: {query}

Answer:"""

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        answer = response.json()["response"].strip()

        return {
            "answer":  answer,
            "sources": [r.chunk.metadata.get("title", "") for r in results],
            "chunks":  [r.chunk.text for r in results],
        }