#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_pipeline.cleanup import purge_stale_data_streaming
from rag_pipeline.retrieval import group_chunk_ids_by_pdf, top_k_indices_desc
from rag_pipeline.storage import save_json_atomic


@dataclass
class MockChunk:
    id: str
    pdf_name: str
    file_hash: str


class MutableCollection:
    """Pagination observes deletion just as a live collection does."""

    def __init__(self, rows: list[dict]):
        self.rows = list(rows)
        self.deleted = 0

    def get(self, limit=5000, offset=0, include=None):
        batch = self.rows[offset : offset + limit]
        return {
            "ids": [row["id"] for row in batch],
            "metadatas": [row["meta"] for row in batch],
        }

    def delete(self, ids):
        removed = set(ids)
        before = len(self.rows)
        self.rows = [r for r in self.rows if r["id"] not in removed]
        self.deleted += before - len(self.rows)


def measure(fn, repeats):
    timings = []
    result = fn()  # unmeasured warmup
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        timings.append((time.perf_counter() - start) * 1000)
    return result, {
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(float(np.percentile(timings, 95)), 3),
    }


def run_benchmark(samples: int, output: Path | None, repeats: int = 5) -> dict:
    if samples <= 0 or repeats <= 0:
        raise ValueError("samples and repeats must be positive")
    scores = np.random.default_rng(42).random(samples)
    chunks = [
        MockChunk(f"doc-{i}", f"book-{i % 100}.pdf", f"h{i % 7}")
        for i in range(samples)
    ]
    rows = [
        {"id": f"id-{i}", "meta": {"source": f"book-{i % 120}.pdf"}}
        for i in range(samples)
    ]
    valid = {f"book-{i}.pdf" for i in range(100)}
    expected_orphans = sum(row["meta"]["source"] not in valid for row in rows)

    def full_sort():
        return np.lexsort((np.arange(samples), -scores))[: min(50, samples)]

    idx, topk = measure(lambda: top_k_indices_desc(scores, 50), repeats)
    reference, sorting = measure(full_sort, repeats)
    if not np.array_equal(idx, reference):
        raise AssertionError("Top-k results differ from full-sort reference")
    grouped, grouping = measure(lambda: group_chunk_ids_by_pdf(chunks), repeats)

    def cleanup():
        collection = MutableCollection(rows)
        deleted = purge_stale_data_streaming(
            collection, valid, fetch_size=5000, delete_batch_size=1000
        )
        if deleted != expected_orphans or any(
            r["meta"]["source"] not in valid for r in collection.rows
        ):
            raise AssertionError("Cleanup skipped or miscounted orphan rows")
        return deleted

    deleted, purge = measure(cleanup, repeats)
    tracemalloc.start()
    cleanup()
    _, cleanup_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "samples": samples,
        "repeats": repeats,
        "seed": 42,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "backend": "mutable in-memory pagination simulator (not Chroma/model throughput)",
        "top_k": topk,
        "full_sort_reference": sorting,
        "group_ids": grouping,
        "purge_orphans": purge,
        "grouped_pdf_count": len(grouped),
        "orphans_deleted": deleted,
        "expected_orphans": expected_orphans,
        "cleanup_peak_python_bytes": cleanup_peak,
        "memory_scope": "Python allocations during cleanup, including simulator copies; excludes input corpus and native allocations",
        "correctness_verified": True,
    }
    if output is not None:
        save_json_atomic(result, output)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark pipeline helpers with correctness checks."
    )
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/latest.json"))
    args = parser.parse_args()
    run_benchmark(args.samples, args.output, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
