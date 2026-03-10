#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_pipeline.cleanup import purge_stale_data_streaming
from rag_pipeline.retrieval import group_chunk_ids_by_pdf, top_k_indices_desc


@dataclass
class MockChunk:
    id: str
    pdf_name: str
    file_hash: str


class FakeCollection:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.deleted = 0

    def get(self, limit=5000, offset=0, include=None):
        batch = self.rows[offset : offset + limit]
        return {
            "ids": [row["id"] for row in batch],
            "metadatas": [row["meta"] for row in batch],
        }

    def delete(self, ids):
        self.deleted += len(ids)


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000.0
    return out, duration_ms


def run_benchmark(samples: int, output: Path | None) -> dict:
    np.random.seed(42)
    scores = np.random.rand(samples)

    chunks = [
        MockChunk(id=f"doc-{i}", pdf_name=f"book-{i % 100}.pdf", file_hash=f"h{i % 7}")
        for i in range(samples)
    ]

    rows = [
        {"id": f"id-{i}", "meta": {"source": f"book-{i % 120}.pdf"}}
        for i in range(samples)
    ]
    collection = FakeCollection(rows)

    tracemalloc.start()

    _, topk_ms = timed(top_k_indices_desc, scores, 50)
    grouped, group_ms = timed(group_chunk_ids_by_pdf, chunks)
    deleted, purge_ms = timed(
        purge_stale_data_streaming,
        collection,
        valid_sources={f"book-{i}.pdf" for i in range(100)},
        fetch_size=5000,
        delete_batch_size=1000,
        logger=None,
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "samples": samples,
        "top_k_ms": round(topk_ms, 3),
        "group_ids_ms": round(group_ms, 3),
        "purge_orphans_ms": round(purge_ms, 3),
        "grouped_pdf_count": len(grouped),
        "orphans_deleted": deleted,
        "memory_current_bytes": current,
        "memory_peak_bytes": peak,
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark key pipeline helper paths.")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/latest.json"))
    args = parser.parse_args()
    run_benchmark(samples=args.samples, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
