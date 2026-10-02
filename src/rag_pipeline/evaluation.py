"""Reproducible retrieval evaluation using independent, explicit judgments."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from statistics import mean, median

from .index import Index
from .retrieval_metrics import evaluate_rankings


def _percentile(samples, fraction):
    values = sorted(samples)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (
        position - lower
    )


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def span_metrics(predicted, relevant):
    """Character coverage with overlap counted once; same-document intersections."""
    docs = {s["document"] for s in predicted + relevant}
    retrieved = labeled = overlap = 0
    for doc in docs:
        found = _union(
            [(s["start"], s["end"]) for s in predicted if s["document"] == doc]
        )
        gold = _union(
            [(s["start"], s["end"]) for s in relevant if s["document"] == doc]
        )
        retrieved += sum(b - a for a, b in found)
        labeled += sum(b - a for a, b in gold)
        overlap += sum(max(0, min(b, d) - max(a, c)) for a, b in found for c, d in gold)
    if not labeled:
        raise ValueError("Span evaluation requires positive relevance spans")
    return {
        "character_recall": overlap / labeled,
        "character_precision": overlap / retrieved if retrieved else 0,
        "retrieved_chars": retrieved,
        "relevant_chars": labeled,
    }


def evaluate(
    dataset: str | Path,
    *,
    mode="lexical",
    embedder=None,
    k=5,
    repeats=3,
    split=None,
    reranker=None,
):
    if not 1 <= repeats <= 100:
        raise ValueError("Repeats must be between 1 and 100")
    data = json.loads(Path(dataset).read_text(encoding="utf-8"))
    docs, queries = data["documents"], data["queries"]
    if len({d["id"] for d in docs}) != len(docs) or len(
        {q["id"] for q in queries}
    ) != len(queries):
        raise ValueError("Duplicate document or query ids")
    known = {d["id"] for d in docs}
    lengths = {d["id"]: len(d["text"]) for d in docs}
    for query in queries:
        for span in query.get("spans", []):
            if (
                span["document"] not in lengths
                or not 0 <= span["start"] < span["end"] <= lengths[span["document"]]
            ):
                raise ValueError("Invalid annotated source span")
    if any(not set(q["relevant"]) <= known for q in queries):
        raise ValueError("Unknown relevance id")
    queries = [q for q in queries if split is None or q.get("split") == split]
    if not queries:
        raise ValueError("No evaluation queries selected")
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as directory:
        root = Path(directory)
        corpus = root / "corpus"
        corpus.mkdir()
        filenames = {}
        for i, doc in enumerate(docs):
            filename = f"{i:08d}.txt"
            filenames[filename] = doc["id"]
            (corpus / filename).write_text(doc["text"], encoding="utf-8")
            (corpus / (filename + ".meta.json")).write_text(
                json.dumps(doc.get("metadata", {}))
            )
        with Index(root / "index.sqlite", embedder, reranker) as index:
            started = time.perf_counter()
            stats = index.ingest(corpus)
            index_seconds = time.perf_counter() - started
            rankings, samples, negative_correct, spans = {}, [], 0, []
            # One global warm-up; model loading is external, index encoding included.
            first = queries[0]
            index.search(
                first["question"], filters=first.get("filters"), mode=mode, k=k
            )
            for query in queries:
                for _ in range(repeats):
                    started = time.perf_counter()
                    hits = index.search(
                        query["question"], filters=query.get("filters"), mode=mode, k=k
                    )
                    samples.append((time.perf_counter() - started) * 1000)
                # Judgment unit is document, not chunk: deduplicate explicitly.
                ranking = list(dict.fromkeys(filenames[h.document] for h in hits))
                rankings[query["id"]] = ranking
                negative_correct += not query["relevant"] and not ranking
                if query.get("spans"):
                    spans.append(
                        span_metrics(
                            [
                                {
                                    "document": filenames[h.document],
                                    "start": h.start,
                                    "end": h.end,
                                }
                                for h in hits
                            ],
                            query["spans"],
                        )
                    )
            qrels = {q["id"]: set(q["relevant"]) for q in queries if q["relevant"]}
            negatives = sum(not q["relevant"] for q in queries)
            return {
                "dataset": data.get("name", str(dataset)),
                "source": data.get("source"),
                "source_sha256": data.get("source_sha256"),
                "skipped_query_ids": data.get("skipped_query_ids", []),
                "limitations": data.get("limitations"),
                "mode": mode,
                "embedding": stats["embedding"],
                "reranker": getattr(reranker, "signature", None),
                "split": split,
                "documents": len(docs),
                "chunks": stats["chunks"],
                "retrieval": evaluate_rankings(rankings, qrels, k) if qrels else None,
                "span_retrieval": {
                    "queries": len(spans),
                    **{
                        metric: mean(s[metric] for s in spans)
                        for metric in ("character_recall", "character_precision")
                    },
                }
                if spans
                else None,
                "negative_queries": negatives,
                "negative_abstention_rate": negative_correct / negatives
                if negatives
                else None,
                "index_seconds": index_seconds,
                "query_samples": len(samples),
                "latency_ms": {
                    "median": median(samples),
                    "p95": _percentile(samples, 0.95),
                    "max": max(samples),
                },
                "rankings": rankings,
            }
