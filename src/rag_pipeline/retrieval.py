from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def top_k_indices_desc(values: np.ndarray, k: int) -> np.ndarray:
    if values is None:
        return np.array([], dtype=int)
    n = int(values.shape[0]) if hasattr(values, "shape") else len(values)
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(k, n)
    idx = np.argpartition(values, -k)[-k:]
    return idx[np.argsort(values[idx])[::-1]]


def reciprocal_rank_fusion(
    dense_ids: Sequence[str], bm25_ids: Sequence[str], k: int = 60
) -> dict[str, float]:
    def rrf(rank: int) -> float:
        return 1.0 / (k + rank + 1)

    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + rrf(rank)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + rrf(rank)
    return scores


def group_chunk_ids_by_pdf(chunks: Iterable[object]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        pdf_name = getattr(chunk, "pdf_name", None)
        chunk_id = getattr(chunk, "id", None)
        if pdf_name and chunk_id:
            grouped[pdf_name].add(chunk_id)
    return dict(grouped)


def first_hash_by_pdf(chunks: Iterable[object]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for chunk in chunks:
        pdf_name = getattr(chunk, "pdf_name", None)
        file_hash = getattr(chunk, "file_hash", None)
        if pdf_name and file_hash and pdf_name not in hashes:
            hashes[pdf_name] = file_hash
    return hashes
