import re
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from athena.core import Chunk, RetrievalResult

class BM25Retriever:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.bm25 = None

    def index(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built: {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in top_indices if scores[i] > 0
        ]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def save(self, path: str = "data/bm25_chunks.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [{"text": c.text, "metadata": c.metadata, "chunk_id": c.chunk_id}
                for c in self.chunks]
        Path(path).write_text(json.dumps(data))

    def load(self, path: str = "data/bm25_chunks.json"):
        data = json.loads(Path(path).read_text())
        chunks = [Chunk(text=d["text"], metadata=d["metadata"], chunk_id=d["chunk_id"])
                  for d in data]
        self.index(chunks)