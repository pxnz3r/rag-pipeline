"""Reproduce durable 100k-row ingest and broad/selective lexical query timings."""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from rag_pipeline import Index


def measure(index, match, repeats):
    samples = []
    for _ in range(repeats + 1):
        started = time.perf_counter()
        hits = index.search("revenue 99123", k=5, match=match)
        samples.append((time.perf_counter() - started) * 1000)
        assert hits and "99123" in hits[0].text
    return {
        "match": match,
        "returned": len(hits),
        "samples_ms": samples[1:],
        "median_ms": float(np.median(samples[1:])),
        "p95_ms": float(np.percentile(samples[1:], 95)),
    }


def benchmark(rows, repeats):
    if rows < 99124 or repeats < 1:
        raise ValueError("At least 99124 rows and one repetition required")
    with tempfile.TemporaryDirectory(prefix="rag-scale-") as directory:
        root = Path(directory)
        corpus = root / "data"
        corpus.mkdir()
        with (corpus / "ledger.csv").open("w") as stream:
            stream.write("account,description,USD,year\n")
            for i in range(rows):
                stream.write(f"{i},annual revenue account {i},125.00,2024\n")
        with Index(root / "index.sqlite") as index:
            started = time.perf_counter()
            stats = index.ingest(corpus)
            ingest_seconds = time.perf_counter() - started
            started = time.perf_counter()
            unchanged = index.ingest(corpus)
            unchanged_seconds = time.perf_counter() - started
            assert stats["chunks"] == rows and unchanged["changed"] == 0
            return {
                "rows": rows,
                "ingest_seconds": ingest_seconds,
                "unchanged_ingest_seconds": unchanged_seconds,
                "index_bytes": index.db.execute("PRAGMA page_count").fetchone()[0]
                * index.db.execute("PRAGMA page_size").fetchone()[0],
                "queries": [measure(index, match, repeats) for match in ("any", "all")],
                "limitations": "Synthetic exact-account query; any matches every row and all matches only the target. Different result semantics, not an equivalent speed comparison. Disk/index pages exclude SQLite/model/native peak memory.",
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.rows, args.repeats)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
