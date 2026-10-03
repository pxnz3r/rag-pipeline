"""Paired question-level bootstrap over fixed retrieval judgments, not answer QA.

Run: python scripts/compare_results.py DATASET BASELINE CANDIDATE --metric ndcg_at_k -k 10
"""

import argparse
import json
import random
from pathlib import Path
from statistics import mean

from rag_pipeline.retrieval_metrics import evaluate_rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--metric",
        choices=["recall_at_k", "mrr_at_k", "ndcg_at_k"],
        default="recall_at_k",
    )
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if not 100 <= args.resamples <= 100000 or not 1 <= args.k <= 1000:
        parser.error("Invalid bootstrap sample count or retrieval cutoff")
    dataset = json.loads(args.dataset.read_text())
    a, b = [
        json.loads(p.read_text())["rankings"] for p in (args.baseline, args.candidate)
    ]
    differences = []
    for q in dataset["queries"]:
        if not q["relevant"]:
            continue
        qrels = {q["id"]: set(q["relevant"])}
        if q["id"] not in a or q["id"] not in b:
            raise ValueError("Both results must cover every positive dataset query")
        scores = [
            evaluate_rankings({q["id"]: ranking[q["id"]]}, qrels, args.k)[args.metric]
            for ranking in (a, b)
        ]
        differences.append(scores[1] - scores[0])
    if not differences:
        raise ValueError("No positive questions to compare")
    rng = random.Random(args.seed)
    samples = sorted(
        mean(rng.choices(differences, k=len(differences)))
        for _ in range(args.resamples)
    )
    print(
        json.dumps(
            dict(
                metric=args.metric,
                k=args.k,
                questions=len(differences),
                difference=mean(differences),
                percentile_bootstrap_95ci=[
                    samples[int(args.resamples * 0.025)],
                    samples[int(args.resamples * 0.975) - 1],
                ],
                resamples=args.resamples,
                seed=args.seed,
                limitations="Paired retrieval questions; same corpus/labels required. Not answer accuracy, not a multiplicity-corrected significance claim.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
