from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from .answers import Operand, answer, calculate, groq_generator
from .evaluation import evaluate
from .index import Index


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rag-pipeline",
        description="Local, evidence-first finance/legal retrieval.",
    )
    parser.add_argument(
        "--index", type=Path, default=Path("processed_data/index.sqlite")
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Load the pinned CPU MiniLM ONNX model (optional models extra).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use the optional CPU cross-encoder; increases latency.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser(
        "ingest", help="Atomically synchronize a complete source directory."
    )
    ingest.add_argument("directory", type=Path)
    ingest.add_argument("--size", type=int, default=900)
    ingest.add_argument("--overlap", type=int, default=100)
    for name in ("search", "ask", "calculate"):
        query = sub.add_parser(name)
        query.add_argument("question")
        query.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
        query.add_argument("-k", type=int, default=5)
        query.add_argument(
            "--mode", choices=["lexical", "dense", "hybrid"], default="hybrid"
        )
        query.add_argument("--match", choices=["all", "any"], default="any")
        if name == "ask":
            query.add_argument(
                "--generate",
                action="store_true",
                help="Use Groq; default returns exact evidence only.",
            )
        if name == "calculate":
            query.add_argument(
                "--operation",
                required=True,
                choices=["sum", "difference", "ratio", "growth_percent"],
            )
            query.add_argument(
                "--operands",
                type=Path,
                required=True,
                help="JSON list of {source_id, quote, value, unit}.",
            )
    sub.add_parser("status")
    bench = sub.add_parser(
        "evaluate",
        help="Evaluate explicitly judged JSON corpus/queries; includes latency.",
    )
    bench.add_argument("dataset", type=Path)
    bench.add_argument(
        "--mode", choices=["lexical", "dense", "hybrid"], default="lexical"
    )
    bench.add_argument("-k", type=int, default=5)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--split", default=None)
    bench.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        embedder = None
        if args.dense:
            from .embeddings import MiniLM

            embedder = MiniLM()
        reranker = None
        if args.rerank and args.command in {"search", "ask", "calculate", "evaluate"}:
            from .embeddings import CrossEncoder

            reranker = CrossEncoder()
        if args.command == "evaluate":
            result = evaluate(
                args.dataset,
                mode=args.mode,
                embedder=embedder,
                k=args.k,
                repeats=args.repeats,
                split=args.split,
                reranker=reranker,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        else:
            if args.command != "ingest" and not args.index.is_file():
                raise ValueError("Index does not exist; ingest a corpus first")
            with Index(args.index, embedder, reranker) as index:
                if args.command == "ingest":
                    result = index.ingest(
                        args.directory, size=args.size, overlap=args.overlap
                    )
                elif args.command == "status":
                    result = index.status()
                else:
                    filters = {}
                    for spec in args.filter:
                        key, separator, value = spec.partition("=")
                        if not separator or key in filters:
                            raise ValueError("Filters require unique KEY=VALUE pairs")
                        filters[key] = value
                    opts = dict(
                        filters=filters, k=args.k, mode=args.mode, match=args.match
                    )
                    if args.command == "search":
                        result = [
                            hit.to_dict() for hit in index.search(args.question, **opts)
                        ]
                    elif args.command == "ask":
                        result = asdict(
                            answer(
                                index,
                                args.question,
                                generate=groq_generator() if args.generate else None,
                                **opts,
                            )
                        )
                    else:
                        sources = index.search(args.question, **opts)
                        operands = [
                            Operand(**item)
                            for item in json.loads(args.operands.read_text())
                        ]
                        result = calculate(args.operation, operands, sources)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (
        ValueError,
        OSError,
        sqlite3.Error,
        ImportError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Operation failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
