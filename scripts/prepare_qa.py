"""Prepare an untouched FinQA numeric holdout, excluding prior retrieval queries.

Gold data is retained for evaluation, but only report/table inputs are indexed.
Requires an explicit provider config and earlier prepare_finqa.py output.
"""

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

from rag_pipeline import Index
from rag_pipeline.configuration import load_config, provider
from rag_pipeline.sources import row_text

REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
HASHES = {
    "dev": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
    "test": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
}


def prepare(
    output,
    exclude,
    config,
    *,
    test_exclude=None,
    test_seed=20261003,
    exclude_test_reports=False,
    table_labels=False,
):
    if output.exists() and any(output.iterdir()):
        raise ValueError("QA preparation requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    seen = {q["id"] for q in json.loads(exclude.read_text())["queries"]}
    prior = json.loads(test_exclude.read_text()) if test_exclude else {}
    prior_records = [row for rows in prior.values() for row in rows] if prior else []
    test_seen = seen | {row["id"] for row in prior_records}
    prior_reports = {row["filename"] for row in prior_records}
    prior_reports.update(
        doc["id"].split("#", 1)[0]
        for doc in json.loads(exclude.read_text()).get("documents", [])
    )
    records, provenance, mapping, views = {}, {}, {}, {}
    corpus = output / "corpus"
    corpus.mkdir(exist_ok=True)
    for split, count in [("dev", 8), ("test", 24)]:
        url = f"https://raw.githubusercontent.com/czyssrs/FinQA/{REVISION}/dataset/{split}.json"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read(20 * 1024 * 1024 + 1)
        if hashlib.sha256(raw).hexdigest() != HASHES[split]:
            raise ValueError("Pinned dataset digest mismatch")
        eligible = [
            r
            for r in json.loads(raw)
            if r["id"] not in (test_seen if split == "test" else seen)
            and not (
                split == "test"
                and exclude_test_reports
                and r["filename"] in prior_reports
            )
            and isinstance(r["qa"].get("exe_ans"), (int, float))
            and not isinstance(r["qa"]["exe_ans"], bool)
        ]
        selected = random.Random(test_seed if split == "test" else 20261003).sample(
            sorted(eligible, key=lambda r: r["id"]), count
        )
        records[split] = selected
        provenance[split] = dict(
            url=url, sha256=HASHES[split], available=len(eligible), selected=count
        )
        for i, record in enumerate(selected):
            name = f"{split}_{i:03d}.md"
            table = "\n".join(" | ".join(map(str, row)) for row in record["table"])
            if table_labels:
                table = "\n\n".join(
                    row_text(record["table"][0], row) for row in record["table"][1:]
                )
            text = (
                "# Report context\n"
                + "\n".join(record["pre_text"])
                + "\n\n# Original table\n"
                + table
                + "\n\n# Subsequent text\n"
                + "\n".join(record["post_text"])
            )
            view = hashlib.sha256(text.encode()).hexdigest()
            key = (record["filename"], view)
            if key in views:
                mapping[record["id"]] = views[key]
                continue
            views[key] = name
            mapping[record["id"]] = name
            (corpus / name).write_text(text, encoding="utf-8")
            (corpus / (name + ".meta.json")).write_text(
                json.dumps(
                    dict(
                        report=record["filename"],
                        title=record["filename"],
                        source_view=view,
                    )
                ),
                encoding="utf-8",
            )
    (output / "records.json").write_text(json.dumps(records, indent=2))
    (output / "mapping.json").write_text(json.dumps(mapping, indent=2))
    (output / "protocol.json").write_text(
        json.dumps(
            dict(
                provenance=provenance,
                dev_seed=20261003,
                test_seed=test_seed,
                test_excluded_query_ids=sorted(test_seen),
                test_excluded_report_ids=sorted(prior_reports)
                if exclude_test_reports
                else [],
                indexed_unique_views=len(views),
                table_rendering="repeated original column labels per row"
                if table_labels
                else "raw matrix with one header row",
                excluded_query_ids=sorted(seen),
                policy="8 development, 24 unseen numeric test cases; report scope is input; no gold in indexing or generation. Numeric execution subset, not official full-test or SOTA score.",
            ),
            indent=2,
        )
    )
    settings = load_config(config)
    with Index(
        output / "index.sqlite", provider(settings["embedding"], "embedding")
    ) as index:
        print(json.dumps(index.ingest(corpus, size=600, overlap=80, contextual=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--test-exclude",
        type=Path,
        help="Earlier records.json; excluded only from the test selection",
    )
    parser.add_argument("--test-seed", type=int, default=20261003)
    parser.add_argument("--exclude-test-reports", action="store_true")
    parser.add_argument("--table-labels", action="store_true")
    args = parser.parse_args()
    prepare(
        args.output,
        args.exclude,
        args.config,
        test_exclude=args.test_exclude,
        test_seed=args.test_seed,
        exclude_test_reports=args.exclude_test_reports,
        table_labels=args.table_labels,
    )
