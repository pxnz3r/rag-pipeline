import numpy as np

from rag_pipeline.retrieval import reciprocal_rank_fusion, top_k_indices_desc


def test_top_k_indices_desc_returns_descending_indices():
    values = np.asarray([0.1, 4.2, 1.0, 8.7, 2.5], dtype=float)
    idx = top_k_indices_desc(values, 3)
    assert idx.tolist() == [3, 1, 4]


def test_top_k_indices_desc_handles_empty():
    values = np.asarray([], dtype=float)
    idx = top_k_indices_desc(values, 5)
    assert idx.size == 0


def test_reciprocal_rank_fusion_combines_both_lists():
    dense = ["a", "b", "c"]
    bm25 = ["c", "a", "x"]
    scores = reciprocal_rank_fusion(dense, bm25, k=60)
    assert set(scores.keys()) == {"a", "b", "c", "x"}
    assert scores["a"] > scores["b"]
