"""Pinned CUAD positive test clauses, original character spans, 10 seed-42 contracts.

Attribution: CUAD by The Atticus Project, Dan Hendrycks et al. (2021), CC BY 4.0.
https://www.atticusprojectai.org/cuad/ ; https://creativecommons.org/licenses/by/4.0/
Changes: deterministic subset and evaluation-format conversion; no text changes.
Run: python scripts/prepare_cuad.py /tmp/cuad-proxy.json
"""

import hashlib
import io
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

REVISION = "67faa0e6023b04fcaae6cc09497ab00e5d63a2a2"
URL = f"https://raw.githubusercontent.com/TheAtticusProject/cuad/{REVISION}/data.zip"


def prepare(output):
    with urllib.request.urlopen(URL, timeout=60) as response:
        raw = response.read(32 * 1024**2 + 1)
    if len(raw) > 32 * 1024**2:
        raise ValueError("Dataset exceeds download limit")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.getinfo("test.json").file_size > 20 * 1024**2:
            raise ValueError("Dataset exceeds expansion limit")
        data = json.loads(archive.read("test.json"))
    documents, queries = [], []
    for record in random.Random(42).sample(data["data"], 10):
        docid = record["title"]
        if len(record["paragraphs"]) != 1:
            raise ValueError("Expected one complete contract per record")
        paragraph = record["paragraphs"][0]
        context = paragraph["context"]
        documents.append(
            {
                "id": docid,
                "text": context,
                "metadata": {"contract_id": docid, "title": docid.rsplit("_", 1)[-1]},
            }
        )
        for query in paragraph["qas"]:
            if not query["answers"]:
                continue
            spans = []
            for answer in query["answers"]:
                start, end = (
                    answer["answer_start"],
                    answer["answer_start"] + len(answer["text"]),
                )
                if context[start:end] != answer["text"]:
                    raise ValueError("Annotation does not match the original source")
                if start < end:
                    spans.append({"document": docid, "start": start, "end": end})
            if spans:
                queries.append(
                    {
                        "id": query["id"],
                        "question": query["question"],
                        "filters": {"contract_id": docid},
                        "relevant": [docid],
                        "spans": spans,
                        "split": "heldout",
                    }
                )
    result = {
        "name": "cuad-test-span-proxy-10-contracts-seed42",
        "source": URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "license": "CC BY 4.0; The Atticus Project, Dan Hendrycks et al., 2021",
        "limitations": "All positive questions from 10 seed-42 sampled test contracts; no training/tuning. Original wording/text/spans. Explicit contract_id scope matches the original this-contract question context. Document recall is trivial under this filter: evaluate character recall/precision. No-answer clauses excluded; NOT official CUAD extraction F1 or full LegalBench-RAG results.",
        "documents": documents,
        "queries": queries,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contracts": len(documents), "questions": len(queries)}))


if __name__ == "__main__":
    prepare(sys.argv[1])
