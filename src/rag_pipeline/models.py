from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Chunk:
    id: str
    text: str
    pdf_name: str
    file_hash: str
    pipeline_version: str
    page_number: int
    chunk_index: int
    pdf_path: str = ""
    char_count: int = 0
    word_count: int = 0
    has_numbers: bool = False
    has_formula: bool = False
    text_with_context: Optional[str] = None
    is_garbled: bool = False

    def to_dict(self):
        return asdict(self)
