"""Prepare the full BEIR SciFact corpus/test retrieval judgments, without training.

Attribution: Wadden et al., Fact or Fiction: Verifying Scientific Claims (EMNLP
2020), https://github.com/allenai/scifact. Claims/annotations: CC BY 4.0;
abstracts: S2ORC ODC-By 1.0. BEIR conversion: Thakur et al. (NeurIPS 2021).
Raw text stays outside the repository. This measures retrieval, not medical
advice accuracy or scientific claim verification.
"""

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


def prepare(raw):
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise ValueError("SciFact archive differs from the reviewed release")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        docs = [
            json.loads(line)
            for line in archive.read("scifact/corpus.jsonl").splitlines()
        ]
        queries = {
            q["_id"]: q
            for q in map(json.loads, archive.read("scifact/queries.jsonl").splitlines())
        }
        relevant = defaultdict(list)
        for row in csv.DictReader(
            io.StringIO(archive.read("scifact/qrels/test.tsv").decode()), delimiter="\t"
        ):
            if int(row["score"]) > 0:
                relevant[row["query-id"]].append(row["corpus-id"])
    return {
        "name": "BEIR SciFact full corpus / all 300 test retrieval queries",
        "source": URL,
        "source_sha256": digest,
        "limitations": "Biomedical abstract retrieval, not clinical QA or support/refute verification. Full 5,183-document corpus; binary positive judgments. Title metadata is limited to 512 characters. No label-based corpus filtering or model tuning.",
        "documents": [
            {"id": d["_id"], "text": d["text"], "metadata": {"title": d["title"][:512]}}
            for d in docs
        ],
        "queries": [
            {
                "id": qid,
                "question": queries[qid]["text"],
                "relevant": ids,
                "split": "test",
            }
            for qid, ids in sorted(relevant.items())
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--archive", type=Path, help="Use a previously downloaded BEIR ZIP"
    )
    args = parser.parse_args()
    if args.archive:
        raw = args.archive.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("Archive exceeds download limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prepare(raw), ensure_ascii=False) + "\n")
