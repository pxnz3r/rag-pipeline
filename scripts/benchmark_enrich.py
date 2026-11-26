"""Measure LLM chunk context (``ingest --enrich``) on a judged dataset subset.

    python scripts/benchmark_enrich.py /tmp/cuad.json --docs 4 \
        --chat-endpoint http://127.0.0.1:8081/v1 --chat-model qwen3.5-4b \
        --embedder arctic-m-v1.5-q --replies /tmp/enrich-replies.sqlite \
        --output enrich.jsonl

Evaluates each mode with and without enrichment on the first ``--docs``
documents and the queries judged only against them. Model replies are kept in
``--replies`` keyed by the exact prompt, so reruns and other modes reuse them
instead of calling the model again.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_beir import models  # noqa: E402

from rag_pipeline.enrich import ChatEnricher  # noqa: E402
from rag_pipeline.evaluation import evaluate  # noqa: E402
from rag_pipeline.providers import ChatServer  # noqa: E402


class Replies:
    """Chat calls memoized by prompt in SQLite; safe across enrichment threads."""

    def __init__(self, chat, path):
        self.chat, self.model = chat, chat.model
        self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS replies(key TEXT PRIMARY KEY, reply TEXT NOT NULL)"
        )
        self.calls = 0

    def __call__(self, system, user):
        key = hashlib.sha256(f"{self.model}\x1f{system}\x1f{user}".encode()).hexdigest()
        row = self.db.execute("SELECT reply FROM replies WHERE key=?", [key]).fetchone()
        if row:
            return row[0]
        reply = self.chat(system, user)
        self.calls += 1
        self.db.execute("INSERT OR REPLACE INTO replies VALUES(?,?)", [key, reply])
        return reply


def subset(data, count):
    documents = data["documents"][:count]
    known = {d["id"] for d in documents}
    queries = [
        q for q in data["queries"] if q["relevant"] and set(q["relevant"]) <= known
    ]
    return {**data, "documents": documents, "queries": queries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--docs", type=int, default=4)
    parser.add_argument("--chat-endpoint", required=True)
    parser.add_argument("--chat-model", required=True)
    parser.add_argument("--embedder", default="none")
    parser.add_argument("--modes", default="lexical,hybrid")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--replies", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = subset(json.loads(args.dataset.read_text(encoding="utf-8")), args.docs)
    chat = Replies(
        ChatServer(
            args.chat_endpoint,
            model=args.chat_model,
            max_tokens=120,
            json_mode=False,
            timeout=600,
        ),
        args.replies,
    )
    # One worker by default: llama-server then reuses the cached prompt prefix
    # (system prompt and document excerpt) across a section's chunks.
    enricher = ChatEnricher(chat, model=args.chat_model, workers=args.workers)
    embedder = None if args.embedder == "none" else models()[args.embedder](4)
    search = {"fusion": "relative", "dense_weight": 0.6}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{args.dataset.stem}-{args.docs}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        for mode in args.modes.split(","):
            for enrich in (None, enricher):
                started, calls = time.perf_counter(), chat.calls
                result = evaluate(
                    path,
                    mode=mode,
                    embedder=embedder if mode != "lexical" else None,
                    k=args.k,
                    repeats=1,
                    enricher=enrich,
                    search_options=search if mode == "hybrid" else None,
                )
                spans = result["span_retrieval"]
                row = {
                    "dataset": args.dataset.stem,
                    "documents": len(data["documents"]),
                    "queries": len(data["queries"]),
                    "chunks": result["chunks"],
                    "mode": mode,
                    "embedder": args.embedder if mode != "lexical" else "none",
                    "enrichment": args.chat_model if enrich else None,
                    f"recall@{args.k}": round(result["retrieval"]["recall_at_k"], 4),
                    f"ndcg@{args.k}": round(result["retrieval"]["ndcg_at_k"], 4),
                    "character_recall": round(spans["character_recall"], 4)
                    if spans
                    else None,
                    "model_calls": chat.calls - calls,
                    "seconds": round(time.perf_counter() - started, 1),
                }
                print(json.dumps(row), flush=True)
                if args.output:
                    with args.output.open("a") as stream:
                        stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
