import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from athena.ingestion.loaders.pdf_loader import PDFLoader
from athena.ingestion.chunkers.recursive import RecursiveChunker
from athena.ingestion.chunkers.metadata_aware import MetadataAwareWrapper
from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from tqdm import tqdm

metadata = json.loads(Path("data/corpus/metadata.json").read_text())

print("Loading embedder (first run downloads BAAI/bge-m3 ~560MB)...")
embedder = BGEEmbedder()
store = QdrantStore()
loader = PDFLoader()
chunker = MetadataAwareWrapper(RecursiveChunker(chunk_size=512, overlap=64))

total_chunks = 0

for paper in tqdm(metadata, desc="Ingesting papers"):
    pdf_path = paper["pdf_path"]
    if not Path(pdf_path).exists():
        continue

    docs = loader.load(pdf_path)
    for doc in docs:
        doc.metadata.update({
            "title":      paper["title"],
            "authors":    paper["authors"],
            "published":  paper["published"],
            "categories": paper["categories"],
            "paper_id":   paper["id"],
        })

    chunks = chunker.chunk(docs)
    if not chunks:
        continue

    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    store.add_chunks(chunks, embeddings)
    total_chunks += len(chunks)

print(f"\nDone. {total_chunks} chunks ingested into Qdrant.")