"""Download a pinned MIT FinQA test slice and prepare a row-retrieval proxy.

Not official FinQA execution evaluation. No gold snippets enter corpus creation.
Run: python scripts/prepare_finqa.py /tmp/finqa-proxy.json
"""

import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
URL = f"https://raw.githubusercontent.com/czyssrs/FinQA/{REVISION}/dataset/test.json"


def prepare(output):
    with urllib.request.urlopen(URL, timeout=60) as response:
        raw = response.read(20 * 1024 * 1024 + 1)
    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("Dataset exceeds download limit")
    records = random.Random(42).sample(json.loads(raw), 100)
    documents, queries, skipped = {}, [], []
    for record in records:
        filename = record["filename"]
        company, year = filename.split("/")[:2]
        columns = record["table"][0]
        entries = {
            f"table_{i}": " ".join(
                f"the {row[0]} of {col} is {v} ;"
                for col, v in zip(columns[1:], row[1:])
            )
            for i, row in enumerate(record["table"])
            if i
        }
        entries.update(
            {
                f"text_{i}": text
                for i, text in enumerate(record["pre_text"] + record["post_text"])
            }
        )
        for row_id, text in entries.items():
            docid = filename + "#" + row_id
            documents[docid] = {
                "id": docid,
                "text": text,
                "metadata": {
                    "company": company,
                    "year": year,
                    "title": company + " " + year,
                },
            }
        relevant = [filename + "#" + key for key in record["qa"]["gold_inds"]]
        if not relevant or not set(relevant) <= documents.keys():
            skipped.append(record["id"])
            continue
        queries.append(
            {
                "id": record["id"],
                "question": record["qa"]["question"],
                "filters": {},
                "relevant": relevant,
                "split": "heldout",
            }
        )
    result = {
        "name": "finqa-test-row-proxy-100-seed42",
        "source": URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "skipped_query_ids": skipped,
        "license": "MIT, Copyright (c) 2021 Zhiyu Chen; see benchmarks/FINQA-LICENSE.txt",
        "limitations": "100 seed-42 sampled test records, all text/table rows as competing documents. Gold row ids are original annotations. Uniform corpus formatting is independent of labels. Excludes questions with absent gold rows. Cross-page row retrieval proxy, NOT official FinQA numerical execution accuracy or full-test results. No question-specific metadata filters, training or relevance tuning.",
        "documents": list(documents.values()),
        "queries": queries,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"documents": len(documents), "queries": len(queries), "skipped": skipped}
        )
    )


if __name__ == "__main__":
    prepare(sys.argv[1])
