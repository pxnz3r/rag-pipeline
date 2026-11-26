"""Prepare FinanceBench (open-source 150 questions) with its source 10-K/10-Q PDFs.

Islam et al., FinanceBench: A New Benchmark for Financial Question Answering
(2023), https://github.com/patronus-ai/financebench (CC BY-NC 4.0). Questions
come from the pinned Hugging Face revision; PDFs from the GitHub repository,
recorded by SHA-256 because that path is not commit-pinned. Raw files stay
outside the repository.

    python scripts/prepare_financebench.py /tmp/fb --cache ~/.cache/rag-pipeline-fb

Writes <output>/corpus/<doc>.pdf with a metadata sidecar and
<output>/financebench.json with questions, gold answers and 1-based evidence
pages (the dataset's evidence_page_num is 0-based).
"""

import argparse
import ast
import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "e04404e3a97f69f79c14d42f24981a1c9c3bcd18"
QUESTIONS = (
    "https://huggingface.co/datasets/PatronusAI/financebench/resolve/"
    f"{REVISION}/financebench_merged.jsonl"
)
PDFS = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{}.pdf"
LIMIT = 64 * 1024 * 1024


def fetch(url, target):
    if not target.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError(f"{url} exceeds the download limit")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache" / "rag-pipeline-fb"
    )
    args = parser.parse_args()
    raw = fetch(QUESTIONS, args.cache / f"questions-{REVISION}.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    docs = sorted({row["doc_name"] for row in rows})
    with ThreadPoolExecutor(8) as pool:
        paths = dict(
            zip(
                docs,
                pool.map(
                    lambda d: fetch(PDFS.format(d), args.cache / "pdfs" / f"{d}.pdf"),
                    docs,
                ),
            )
        )
    corpus = args.output / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    meta = {row["doc_name"]: row for row in rows}
    for doc, path in paths.items():
        if path.read_bytes()[:5] != b"%PDF-":
            raise ValueError(f"{doc} is not a PDF (LFS pointer or error page)")
        shutil.copyfile(path, corpus / f"{doc}.pdf")
        row = meta[doc]
        (corpus / f"{doc}.pdf.meta.json").write_text(
            json.dumps(
                {
                    "company": row["company"],
                    "doc_type": row["doc_type"],
                    "period": str(row["doc_period"]),
                }
            )
        )
    questions = []
    for row in rows:
        evidence = row["evidence"]
        if isinstance(evidence, str):
            evidence = ast.literal_eval(evidence)
        questions.append(
            {
                "id": row["financebench_id"],
                "question": row["question"],
                "answer": row["answer"],
                "justification": row.get("justification"),
                "question_type": row["question_type"],
                "reasoning": row.get("question_reasoning"),
                "document": f"{row['doc_name']}.pdf",
                "company": row["company"],
                "evidence_pages": sorted(
                    {int(e["evidence_page_num"]) + 1 for e in evidence}
                ),
                "evidence_text": [e["evidence_text"] for e in evidence],
            }
        )
    manifest = {
        "name": "FinanceBench open-source 150",
        "source": QUESTIONS,
        "license": "CC BY-NC 4.0",
        "pdf_sha256": {
            f"{d}.pdf": hashlib.sha256(p.read_bytes()).hexdigest()
            for d, p in paths.items()
        },
        "questions": questions,
    }
    (args.output / "financebench.json").write_text(json.dumps(manifest, indent=1))
    print(len(docs), "documents", len(questions), "questions")


if __name__ == "__main__":
    main()
