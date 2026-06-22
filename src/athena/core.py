from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float = 0.0