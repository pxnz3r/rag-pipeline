"""Exact dense scan ablation with deterministic synthetic 384-dimensional vectors.

This isolates storage/top-k selection, not model accuracy or real embedding
inference. Run old/new revisions sequentially under the same BLAS thread settings.
"""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from statistics import median

import numpy as np

from rag_pipeline import Index


class Synthetic:
    signature = "benchmark:synthetic-384:v1"

    def encode(self, texts):
        return np.array(
            [
                np.random.default_rng(
                    int.from_bytes(
                        hashlib.blake2b(text.encode(), digest_size=8).digest(), "little"
                    )
                ).normal(size=384)
                for text in texts
            ],
            dtype=np.float32,
        )


def benchmark(rows, repeats):
    with tempfile.TemporaryDirectory(prefix="rag-dense-") as directory:
        root = Path(directory)
        corpus = root / "corpus"
        corpus.mkdir()
        (corpus / "ledger.csv").write_text(
            "account,description\n"
            + "\n".join(f"{i},Revenue entry {i}" for i in range(rows))
        )
        with Index(root / "index.sqlite", Synthetic()) as index:
            started = time.perf_counter()
            stats = index.ingest(corpus)
            ingest_seconds = time.perf_counter() - started
            question = "Revenue entry 99123"
            options = dict(mode="dense", min_cosine=-1, candidates=40, k=5)
            index.search(question, **options)
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                hits = index.search(question, **options)
                samples.append((time.perf_counter() - started) * 1000)
            return {
                "rows": rows,
                "dimension": 384,
                "stats": stats,
                "ingest_seconds": ingest_seconds,
                "query": question,
                "options": options,
                "latency_samples_ms": samples,
                "median_ms": median(samples),
                "hit_ids": [h.id for h in hits],
                "scores": [h.score for h in hits],
                "limitations": __doc__,
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if not 1 <= args.rows <= 1000000 or not 1 <= args.repeats <= 100:
        parser.error("Require 1..1000000 rows and 1..100 repeats")
    print(json.dumps(benchmark(args.rows, args.repeats), indent=2))
