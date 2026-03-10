from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from .models import Chunk


def tokenize_for_bm25(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"\w+|[$%().,]", text.lower())
    return [t for t in tokens if len(t) > 1 or t in ["$", "%"]]


def build_bm25_index(
    chunks: List[Chunk], logger=None
) -> Tuple[BM25Okapi | None, dict[int, str]]:
    corpus_texts = [
        (c.text_with_context if c.text_with_context else c.text) for c in chunks
    ]
    if not corpus_texts:
        if logger:
            logger.warning("No chunks available for BM25 indexing.")
        return None, {}
    tokenized_corpus = [tokenize_for_bm25(doc) for doc in corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    bm25_id_map = {i: c.id for i, c in enumerate(chunks)}
    return bm25_index, bm25_id_map
