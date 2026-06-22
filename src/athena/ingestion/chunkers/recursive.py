from athena.core import Document, Chunk

class RecursiveChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " "]

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in documents:
            splits = self._split(doc.text)
            for i, text in enumerate(splits):
                if not text.strip():
                    continue
                chunks.append(Chunk(
                    text=text.strip(),
                    metadata={**doc.metadata, "chunk_index": i},
                    chunk_id=f"{doc.doc_id}_recursive_{i}",
                ))
        return chunks

    def _split(self, text: str) -> list[str]:
        for sep in self.separators:
            if sep in text:
                parts = text.split(sep)
                return self._merge(parts, sep)
        return [text]

    def _merge(self, parts: list[str], sep: str) -> list[str]:
        chunks, current, length = [], [], 0
        for part in parts:
            words = len(part.split())
            if length + words > self.chunk_size and current:
                chunks.append(sep.join(current))
                overlap_parts = current[-2:] if self.overlap > 0 else []
                current = overlap_parts + [part]
                length = sum(len(p.split()) for p in current)
            else:
                current.append(part)
                length += words
        if current:
            chunks.append(sep.join(current))
        return chunks