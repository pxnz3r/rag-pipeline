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


def test_topk_ties_are_deterministic_and_nan_excluded():
    values = np.asarray([1, 1, np.nan, 1, -np.inf])
    assert top_k_indices_desc(values, 2).tolist() == [0, 1]
    assert top_k_indices_desc(values, 20).tolist() == [0, 1, 3]


def test_rrf_duplicate_ids_do_not_inflate_scores():
    assert reciprocal_rank_fusion(["a", "a", "b"], ["a"]) == reciprocal_rank_fusion(
        ["a", "b"], ["a"]
    )
