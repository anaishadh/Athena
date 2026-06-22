from athena.core import Document, Chunk

class FixedChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in documents:
            words = doc.text.split()
            step = self.chunk_size - self.overlap
            for i, start in enumerate(range(0, len(words), step)):
                text = " ".join(words[start:start + self.chunk_size])
                if not text.strip():
                    continue
                chunks.append(Chunk(
                    text=text,
                    metadata={**doc.metadata, "chunk_index": i},
                    chunk_id=f"{doc.doc_id}_fixed_{i}",
                ))
        return chunks