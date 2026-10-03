from __future__ import annotations

import argparse
import json
import os
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
        description="Local, evidence-first document and source-code retrieval.",
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
    parser.add_argument(
        "--embedding",
        choices=["minilm", "e5"],
        default="minilm",
        help="Encoder to load with --dense; must match the stored index.",
    )
    parser.add_argument(
        "--reranker",
        choices=["cross-encoder", "colbert", "modern-colbert"],
        default="cross-encoder",
        help="Candidate scoring model to use with --rerank.",
    )
    parser.add_argument(
        "--embedding-server",
        help="Self-hosted OpenAI-compatible /v1 endpoint; use with --dense.",
    )
    parser.add_argument(
        "--server-model", help="Embedding model served by the endpoint."
    )
    parser.add_argument(
        "--server-revision",
        help="Immutable model commit or SHA-256 digest; must match the server artifact.",
    )
    parser.add_argument(
        "--vector-cache-mib",
        type=int,
        default=64,
        help="Memory budget for repeated unfiltered dense queries; 0 streams all vectors.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser(
        "ingest", help="Atomically synchronize a complete source directory."
    )
    ingest.add_argument("directory", type=Path)
    ingest.add_argument("--size", type=int, default=900)
    ingest.add_argument("--overlap", type=int, default=100)
    ingest.add_argument(
        "--contextual",
        action="store_true",
        help="Index extractive Markdown heading context.",
    )
    for name in ("search", "ask", "calculate"):
        query = sub.add_parser(name)
        query.add_argument("question")
        query.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
        query.add_argument("-k", type=int, default=5)
        query.add_argument(
            "--mode", choices=["lexical", "dense", "hybrid"], default="hybrid"
        )
        query.add_argument("--match", choices=["all", "any"], default="any")
        query.add_argument(
            "--distinct-documents",
            action="store_true",
            help="Select one evidence window per document.",
        )
        if name == "ask":
            query.add_argument(
                "--context-order", choices=["ranked", "source"], default="ranked"
            )
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
    sub.add_parser(
        "prepare-reranker",
        help="Explicitly encode passage tokens once for ColBERT candidate reranking.",
    )
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
    bench.add_argument("--contextual", action="store_true")
    bench.add_argument("--distinct-documents", action="store_true")
    bench.add_argument(
        "--cache-reranker",
        action="store_true",
        help="Include explicit offline passage-token encoding before timed queries.",
    )
    args = parser.parse_args()
    try:
        embedder = None
        if (
            args.dense
            and args.command != "status"
            and getattr(args, "mode", None) != "lexical"
        ):
            if args.embedding_server:
                from .providers import EmbeddingServer

                embedder = EmbeddingServer(
                    args.embedding_server,
                    model=args.server_model,
                    revision=args.server_revision,
                    api_key_env="RAG_EMBEDDING_API_KEY"
                    if "RAG_EMBEDDING_API_KEY" in os.environ
                    else None,
                )
            else:
                from .embeddings import E5, MiniLM

                embedder = E5() if args.embedding == "e5" else MiniLM()
        reranker = None
        if (
            args.rerank and args.command in {"search", "ask", "calculate", "evaluate"}
        ) or args.command == "prepare-reranker":
            from .embeddings import ColBERT, CrossEncoder, ModernColBERT

            reranker = {
                "colbert": ColBERT,
                "modern-colbert": ModernColBERT,
                "cross-encoder": CrossEncoder,
            }[args.reranker]()
        if args.command == "evaluate":
            result = evaluate(
                args.dataset,
                mode=args.mode,
                embedder=embedder,
                k=args.k,
                repeats=args.repeats,
                split=args.split,
                reranker=reranker,
                contextual=args.contextual,
                vector_cache_bytes=args.vector_cache_mib * 1024 * 1024,
                cache_reranker=args.cache_reranker,
                distinct_documents=args.distinct_documents,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        else:
            if args.command != "ingest" and not args.index.is_file():
                raise ValueError("Index does not exist; ingest a corpus first")
            with Index(
                args.index,
                embedder,
                reranker,
                vector_cache_bytes=args.vector_cache_mib * 1024 * 1024,
            ) as index:
                if args.command == "ingest":
                    result = index.ingest(
                        args.directory,
                        size=args.size,
                        overlap=args.overlap,
                        contextual=args.contextual,
                    )
                elif args.command == "status":
                    result = index.status()
                elif args.command == "prepare-reranker":
                    result = index.prepare_reranker_cache()
                else:
                    filters = {}
                    for spec in args.filter:
                        key, separator, value = spec.partition("=")
                        if not separator or key in filters:
                            raise ValueError("Filters require unique KEY=VALUE pairs")
                        filters[key] = value
                    opts = dict(
                        filters=filters,
                        k=args.k,
                        mode=args.mode,
                        match=args.match,
                        distinct_documents=args.distinct_documents,
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
                                context_order=args.context_order,
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
