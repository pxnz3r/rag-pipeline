"""FinanceBench evidence-page retrieval for a preset: does a gold page come back?

    python scripts/prepare_financebench.py /tmp/fb
    python scripts/benchmark_financebench.py /tmp/fb --preset hybrid --work /tmp/fb/work

Scopes: "shared" searches all 84 filings at once (the hard, realistic setting);
"document" filters to the question's filing (FinanceBench's single-document
setting). A hit counts when its document and page locator match a gold
evidence page. Appends one JSON line per scope to --output.
"""

import argparse
import json
import time
from pathlib import Path
from statistics import median

from rag_pipeline import Index
from rag_pipeline.configuration import provider
from rag_pipeline.presets import config


def page(locator):
    kind, _, number = locator.partition(" ")
    return int(number) if kind == "page" else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--preset", default="lexical")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--parser", default="pypdf")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings = config(args.preset)
    embedder = (
        provider(settings["embedding"], "embedding")
        if "embedding" in settings
        else None
    )
    reranker = (
        provider(settings["reranker"], "reranker") if "reranker" in settings else None
    )
    search = {**settings.get("search", {}), "k": args.k}
    search.update({k: json.loads(v) for k, v in (o.split("=", 1) for o in args.option)})
    manifest = json.loads((args.data / "financebench.json").read_text())
    encoder = settings.get("embedding", {}).get("factory", "none").split(":")[-1]
    path = args.work / f"fb-{args.parser}-{encoder}.sqlite"
    args.work.mkdir(parents=True, exist_ok=True)
    with Index(path, embedder, reranker) as index:
        started = time.perf_counter()
        stats = index.ingest(args.data / "corpus")
        ingest_seconds = time.perf_counter() - started
        for scope in ("shared", "document"):
            found, reciprocal, samples, per_question = 0, 0.0, [], []
            for q in manifest["questions"]:
                filters = (
                    {"title": Path(q["document"]).stem} if scope == "document" else None
                )
                started = time.perf_counter()
                hits = index.search(q["question"], filters=filters, **search)
                samples.append((time.perf_counter() - started) * 1000)
                gold = {(q["document"], p) for p in q["evidence_pages"]}
                ranks = [
                    i
                    for i, h in enumerate(hits, 1)
                    if (h.document, page(h.locator)) in gold
                ]
                found += bool(ranks)
                reciprocal += 1 / ranks[0] if ranks else 0
                per_question.append(
                    {
                        "id": q["id"],
                        "rank": ranks[0] if ranks else None,
                        "hits": [[h.document, h.locator] for h in hits],
                    }
                )
            n = len(manifest["questions"])
            row = {
                "benchmark": "financebench-evidence-pages",
                "preset": args.preset,
                "parser": args.parser,
                "scope": scope,
                "k": args.k,
                "search": search,
                "documents": stats["documents"],
                "chunks": stats["chunks"],
                "ingest_seconds": round(ingest_seconds, 1),
                f"page_recall@{args.k}": round(found / n, 4),
                f"page_mrr@{args.k}": round(reciprocal / n, 4),
                "query_ms_median": round(median(samples), 1),
                "questions": per_question,
            }
            print(
                json.dumps({k: v for k, v in row.items() if k != "questions"}),
                flush=True,
            )
            if args.output:
                with args.output.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
