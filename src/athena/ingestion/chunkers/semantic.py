import spacy
import numpy as np
from athena.core import Document, Chunk
from athena.ingestion.embedders.bge import BGEEmbedder

class SemanticChunker:
    """Splits documents at points where embedding similarity drops sharply.
    
    Unlike fixed/recursive chunkers that split on character count or punctuation,
    semantic chunking detects topic boundaries in the actual meaning of the text.
    This produces chunks that are internally coherent — each chunk covers one idea.
    
    Tradeoff: much slower than fixed/recursive (requires embedding every sentence),
    but produces higher quality chunks for retrieval.
    """

    def __init__(self, embedder: BGEEmbedder,
                 breakpoint_threshold: float = 0.82,
                 min_chunk_words: int = 80,
                 max_chunk_words: int = 500):
        self.embedder  = embedder
        self.threshold = breakpoint_threshold
        self.min_words = min_chunk_words
        self.max_words = max_chunk_words
        self.nlp       = spacy.load("en_core_web_sm")

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in documents:
            try:
                sentences = [s.text.strip()[:1000] for s in self.nlp(doc.text).sents
                             if 20 < len(s.text.strip()) < 2000]
                if not sentences:
                    continue

                embeddings = self.embedder.embed(sentences)
                groups     = self._group_sentences(sentences, embeddings)

                for i, text in enumerate(groups):
                    if not text.strip():
                        continue
                    chunks.append(Chunk(
                        text=text,
                        metadata={**doc.metadata, "chunk_index": i,
                                  "chunker": "semantic"},
                        chunk_id=f"{doc.doc_id}_sem_{i}",
                    ))
            except Exception as e:
                print(f"  ⚠ Skipping doc {doc.doc_id}: {e}")
                continue
        return chunks

    def _group_sentences(self, sentences: list[str],
                          embeddings: list) -> list[str]:
        groups, current = [], [sentences[0]]

        for i in range(1, len(sentences)):
            sim = np.dot(embeddings[i - 1], embeddings[i])
            current_words = sum(len(s.split()) for s in current)

            if (sim < self.threshold and current_words >= self.min_words) \
                    or current_words >= self.max_words:
                groups.append(" ".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        if current:
            groups.append(" ".join(current))
        return groups