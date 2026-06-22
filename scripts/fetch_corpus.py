import arxiv
import json
import urllib.request
from pathlib import Path
from tqdm import tqdm

QUERIES = [
    ("retrieval augmented generation RAG dense retrieval", 25),
    ("sentence embeddings BERT contrastive learning MTEB", 20),
    ("transformer attention mechanism large language model", 25),
    ("LLM agent tool use reasoning planning ReAct", 20),
    ("vector database approximate nearest neighbor HNSW", 10),
]

OUTPUT_DIR = Path("data/corpus/papers")
METADATA_FILE = Path("data/corpus/metadata.json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = arxiv.Client()
all_metadata = []
seen_ids = set()

for query, max_results in QUERIES:
    print(f"\nFetching: {query}")
    search = arxiv.Search(query=query, max_results=max_results,
                          sort_by=arxiv.SortCriterion.Relevance)
    for paper in tqdm(client.results(search)):
        paper_id = paper.entry_id.split("/")[-1]
        if paper_id in seen_ids:
            continue
        seen_ids.add(paper_id)
        pdf_path = OUTPUT_DIR / f"{paper_id}.pdf"
        if not pdf_path.exists():
            try:
                urllib.request.urlretrieve(paper.pdf_url, pdf_path)
            except Exception as e:
                print(f"  ✗ Failed {paper_id}: {e}")
                continue
        all_metadata.append({
            "id": paper_id,
            "title": paper.title,
            "authors": [str(a) for a in paper.authors],
            "published": str(paper.published.date()),
            "categories": paper.categories,
            "summary": paper.summary,
            "pdf_path": str(pdf_path),
        })
        print(f"  ✓ {paper.title[:70]}")

METADATA_FILE.write_text(json.dumps(all_metadata, indent=2))
print(f"\nDone. {len(all_metadata)} papers saved to {METADATA_FILE}")