from athena.core import Document, Chunk

class MetadataAwareWrapper:
    """Prepends paper metadata to every chunk before embedding.
    
    Based on Anthropic's contextual retrieval finding: prepending context
    improves retrieval recall by 35-49% because chunks become self-contained.
    """

    def __init__(self, base_chunker):
        self.chunker = base_chunker

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks = self.chunker.chunk(documents)
        for chunk in chunks:
            m = chunk.metadata
            parts = []
            if m.get("title"):      parts.append(f"Paper: {m['title']}")
            if m.get("authors"):    parts.append(f"Authors: {', '.join(m['authors'][:2])}")
            if m.get("published"):  parts.append(f"Year: {m['published'][:4]}")
            if m.get("categories"): parts.append(f"Field: {m['categories'][0]}")
            if parts:
                prefix = " | ".join(parts)
                chunk.text = f"[{prefix}]\n\n{chunk.text}"
        return chunks