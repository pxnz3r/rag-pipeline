from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevant_ids: Mapping[str, set[str]],
    k: int = 5,
) -> dict[str, float | int]:
    """Macro recall, precision, MRR, and binary NDCG for explicitly judged queries.

    Missing rankings score zero. Duplicate results are ignored. Empty relevance
    judgments are rejected so a missing label cannot silently inflate metrics.
    These metrics evaluate retrieval, not factuality of generated answers.
    """
    if k <= 0 or not relevant_ids or any(not ids for ids in relevant_ids.values()):
        raise ValueError("Positive k and nonempty relevance judgments are required")
    recall = precision = reciprocal_rank = ndcg = 0.0
    for query, relevant in relevant_ids.items():
        ranked = list(dict.fromkeys(rankings.get(query, [])))[:k]
        hits = [i for i, cid in enumerate(ranked, start=1) if cid in relevant]
        recall += len(hits) / len(relevant)
        precision += len(hits) / k
        reciprocal_rank += 1 / hits[0] if hits else 0
        dcg = sum(1 / math.log2(i + 1) for i in hits)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
        ndcg += dcg / ideal
    count = len(relevant_ids)
    return {
        "queries": count,
        "k": k,
        "recall_at_k": recall / count,
        "precision_at_k": precision / count,
        "mrr_at_k": reciprocal_rank / count,
        "ndcg_at_k": ndcg / count,
    }
