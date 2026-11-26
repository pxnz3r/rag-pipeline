"""FinanceBench answer accuracy for one retrieval preset and one chat model.

    python scripts/benchmark_answers.py /tmp/fb --preset hybrid --work /tmp/fb/work \
        --chat-endpoint http://127.0.0.1:8081/v1 --chat-model qwen3.5-4b \
        --route agent --limit 30 --output answers.jsonl

Routes: "single" retrieves once and asks the model for cited claims; "agent"
lets the model search, read and calculate in steps first. Both use the same
quote and number validation, so an answer either cites source text or fails.

Grading: a question is correct when a number in the gold answer appears in the
response within 1% (or after a thousand/million rescale, since filings mix
units). Questions whose gold answer has no number are graded by the chat model
as judge (--judge), and reported separately.
"""

import argparse
import json
import re
import time
from pathlib import Path

from rag_pipeline import Index, answer
from rag_pipeline.configuration import provider
from rag_pipeline.navigation import Navigation
from rag_pipeline.presets import config

NUMBER = re.compile(r"\(?-?\$?\d[\d,]*\.?\d*\)?%?")


def numbers(text):
    values = []
    for token in NUMBER.findall(text or ""):
        negative = token.startswith("(") or "-" in token
        digits = re.sub(r"[^\d.]", "", token)
        if not digits or digits == ".":
            continue
        try:
            value = float(digits)
        except ValueError:
            continue
        values.append(-value if negative else value)
    return values


def numeric_match(gold, response):
    expected = [
        v
        for v in numbers(gold)
        if v not in (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)
    ]
    if not expected:
        return None
    found = numbers(response)
    for target in expected[:1]:  # the headline figure is the first number
        for value in found:
            for scale in (1, 1e3, 1e-3, 1e6, 1e-6, 100, 0.01):
                candidate = value * scale
                if target == 0 and abs(candidate) < 1e-9:
                    return True
                if target and abs(candidate - target) <= 0.01 * abs(target):
                    return True
    return False


def response_text(result):
    if result.status == "calculated":
        data = json.loads(result.answer)
        return " ".join(str(c.get("result", "")) for c in data["calculations"])
    return result.answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--preset", default="hybrid")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--chat-endpoint", required=True)
    parser.add_argument("--chat-model", required=True)
    parser.add_argument("--route", choices=["single", "agent"], default="single")
    parser.add_argument("--scope", choices=["document", "shared"], default="document")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
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
    search = {**settings.get("search", {}), "k": 5}
    chat = provider(
        {
            "kind": "openai",
            "options": {
                "endpoint": args.chat_endpoint,
                "model": args.chat_model,
                "max_tokens": 1500,
                "timeout": 600,
            },
        },
        "generation",
    )
    encoder = settings.get("embedding", {}).get("factory", "none").split(":")[-1]
    manifest = json.loads((args.data / "financebench.json").read_text())
    questions = manifest["questions"][: args.limit or None]
    done = set()
    if args.output.exists():
        done = {json.loads(line)["id"] for line in args.output.read_text().splitlines()}
    with Index(args.work / f"fb-pdfium-{encoder}.sqlite", embedder, reranker) as index:
        index.ingest(args.data / "corpus")
        for q in questions:
            if q["id"] in done:
                continue
            filters = (
                {"title": Path(q["document"]).stem}
                if args.scope == "document"
                else None
            )
            started = time.perf_counter()
            trace = None
            if args.route == "agent":
                session = Navigation(
                    index,
                    filters=filters,
                    max_calls=args.max_steps,
                    search={k: v for k, v in search.items() if k != "k"},
                )
                result = session.answer(q["question"], plan=chat, generate=chat)
                trace = session.trace
            else:
                result = answer(
                    index, q["question"], filters=filters, generate=chat, **search
                )
            text = response_text(result)
            correct = numeric_match(q["answer"], text)
            judged = None
            if correct is None and args.judge and text:
                verdict = chat(
                    'You grade answers. Reply with JSON {"correct": true} or {"correct": false}.',
                    json.dumps(
                        {
                            "question": q["question"],
                            "gold": q["answer"],
                            "response": text,
                        }
                    ),
                )
                try:
                    reply = verdict[verdict.find("{") : verdict.rfind("}") + 1]
                    judged = json.loads(reply).get("correct") is True
                except ValueError:
                    judged = False
            row = {
                "id": q["id"],
                "route": args.route,
                "preset": args.preset,
                "model": args.chat_model,
                "status": result.status,
                "detail": result.detail,
                "gold": q["answer"],
                "response": text,
                "numeric_correct": correct,
                "judged_correct": judged,
                "cited_pages": sorted({h.locator for h in result.sources}),
                "gold_pages": q["evidence_pages"],
                "seconds": round(time.perf_counter() - started, 1),
                "trace": trace,
            }
            with args.output.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {k: row[k] for k in ("id", "status", "numeric_correct", "seconds")}
                ),
                flush=True,
            )
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    graded = [r for r in rows if r["numeric_correct"] is not None]
    print(
        json.dumps(
            {
                "questions": len(rows),
                "numeric_graded": len(graded),
                "numeric_accuracy": round(
                    sum(r["numeric_correct"] for r in graded) / max(1, len(graded)), 4
                ),
                "statuses": {
                    s: sum(r["status"] == s for r in rows)
                    for s in {r["status"] for r in rows}
                },
            }
        )
    )


if __name__ == "__main__":
    main()
