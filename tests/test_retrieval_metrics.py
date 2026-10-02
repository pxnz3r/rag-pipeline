import pytest

from rag_pipeline.retrieval_metrics import evaluate_rankings


def test_rank_metrics_are_macro_averages_with_missing_rankings():
    result = evaluate_rankings(
        {"q1": ["wrong", "a", "a", "b"]}, {"q1": {"a", "b"}, "q2": {"c"}}, k=3
    )
    assert result["recall_at_k"] == 0.5
    assert result["precision_at_k"] == pytest.approx(1 / 3)
    assert result["mrr_at_k"] == 0.25
    assert 0 < result["ndcg_at_k"] < 0.5


def test_empty_judgments_are_not_silently_counted_as_success():
    with pytest.raises(ValueError):
        evaluate_rankings({}, {"q": set()})
