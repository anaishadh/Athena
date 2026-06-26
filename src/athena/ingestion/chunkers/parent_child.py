from athena.core import Document, Chunk
from athena.ingestion.chunkers.recursive import RecursiveChunker

class ParentChildChunker:
    """Indexes small child chunks for precise retrieval, stores large parent
    chunks in metadata for richer generation context.
    
    The problem it solves: small chunks retrieve precisely but lack context
    for the LLM to generate a good answer. Large chunks have context but
    retrieve imprecisely because the embedding averages over too many topics.
    
    Solution: embed and search the child, but return the parent to the LLM.
    """

    def __init__(self, parent_size: int = 1024, child_size: int = 256,
                 parent_overlap: int = 128, child_overlap: int = 32):
        self.parent_chunker = RecursiveChunker(parent_size, parent_overlap)
        self.child_chunker  = RecursiveChunker(child_size, child_overlap)

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in documents:
            parents = self.parent_chunker.chunk([doc])
            for parent in parents:
                parent_doc = Document(
                    text=parent.text,
                    metadata=parent.metadata,
                    doc_id=parent.chunk_id,
                )
                children = self.child_chunker.chunk([parent_doc])
                for child in children:
                    child.metadata["parent_text"] = parent.text
                    child.metadata["parent_id"]   = parent.chunk_id
                    child.metadata["chunker"]      = "parent_child"
                    chunks.append(child)
        return chunks