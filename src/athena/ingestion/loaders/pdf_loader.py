import fitz
from pathlib import Path
from athena.core import Document

class PDFLoader:
    def load(self, pdf_path: str) -> list[Document]:
        doc = fitz.open(pdf_path)
        documents = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if len(text) < 50:
                continue
            documents.append(Document(
                text=text,
                metadata={
                    "source": Path(pdf_path).name,
                    "page": page_num + 1,
                    "total_pages": len(doc),
                },
                doc_id=f"{Path(pdf_path).stem}_p{page_num + 1}",
            ))
        doc.close()
        return documents