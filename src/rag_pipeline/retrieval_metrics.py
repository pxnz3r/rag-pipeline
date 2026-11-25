from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevant_ids: Mapping[str, set[str]],
    k: int = 5,
    grades: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, float | int]:
    """Macro recall, precision, MRR, and NDCG for explicitly judged queries.

    NDCG uses linear gains from ``grades`` (as trec_eval and BEIR do) and is
    binary without them. Missing rankings score zero. Duplicate results are
    ignored. Empty relevance judgments are rejected so a missing label cannot
    silently inflate metrics. These evaluate retrieval, not generated answers.
    """
    if k <= 0 or not relevant_ids or any(not ids for ids in relevant_ids.values()):
        raise ValueError("Positive k and nonempty relevance judgments are required")
    recall = precision = reciprocal_rank = ndcg = 0.0
    for query, relevant in relevant_ids.items():
        gain = (grades or {}).get(query) or dict.fromkeys(relevant, 1)
        ranked = list(dict.fromkeys(rankings.get(query, [])))[:k]
        hits = [i for i, cid in enumerate(ranked, start=1) if cid in relevant]
        recall += len(hits) / len(relevant)
        precision += len(hits) / k
        reciprocal_rank += 1 / hits[0] if hits else 0
        dcg = sum(
            gain.get(cid, 0) / math.log2(i + 1) for i, cid in enumerate(ranked, 1)
        )
        best = sorted((gain.get(cid, 1) for cid in relevant), reverse=True)[:k]
        ideal = sum(g / math.log2(i + 1) for i, g in enumerate(best, 1))
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
