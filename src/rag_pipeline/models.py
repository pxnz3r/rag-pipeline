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

    def __post_init__(self) -> None:
        for name in ("id", "pdf_name", "file_hash", "pipeline_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.text_with_context is not None and not isinstance(
            self.text_with_context, str
        ):
            raise ValueError("text_with_context must be a string or None")
        for name, minimum in (
            ("page_number", 1),
            ("chunk_index", 0),
            ("char_count", 0),
            ("word_count", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    def to_dict(self):
        return asdict(self)
