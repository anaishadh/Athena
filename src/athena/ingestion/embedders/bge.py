import torch
from sentence_transformers import SentenceTransformer

class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
            device=self.device,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed([prefixed])[0]