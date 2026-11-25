"""Prepare a full BEIR test set (corpus, queries, graded judgments) from pinned MTEB mirrors.

Sources are the MTEB Hugging Face copies of the BEIR releases (Thakur et al.,
NeurIPS 2021), fetched at an immutable commit. Licenses stay with the original
datasets: SciFact (CC BY 4.0 claims, ODC-By abstracts), NFCorpus (research use,
Boteva et al. ECIR 2016), FiQA-2018 (Maia et al., WWW 2018 challenge terms).
Raw text stays outside the repository. Nothing here trains or tunes on labels.
"""

import argparse
import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

DATASETS = {
    "scifact": "cf10ab6856b15b0e670ef8ae5dae4e266c12d035",
    "nfcorpus": "52ac3f19d3449632d9f00aab0ad34a110fc03816",
    "fiqa": "5e59eeb3a7df6b85882112b747008547c21587ea",
}
LIMIT = 256 * 1024 * 1024


def fetch(name, revision, path, cache):
    target = cache / name / revision / path
    if not target.exists():
        url = f"https://huggingface.co/datasets/mteb/{name}/resolve/{revision}/{path}"
        with urllib.request.urlopen(url, timeout=120) as response:
            raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError(f"{path} exceeds the download limit")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return target.read_bytes()


def prepare(name, cache):
    revision = DATASETS[name]
    files = {
        p: fetch(name, revision, p, cache)
        for p in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
    }
    docs = [
        json.loads(line) for line in files["corpus.jsonl"].splitlines() if line.strip()
    ]
    queries = {
        q["_id"]: q["text"]
        for q in map(json.loads, files["queries.jsonl"].splitlines())
        if q
    }
    grades = defaultdict(dict)
    lines = files["qrels/test.tsv"].decode().splitlines()
    for line in lines[1:]:
        query, doc, score = line.split("\t")
        if int(score) > 0:
            grades[query][doc] = int(score)
    docs = [d for d in docs if d["text"].strip()]
    known = {d["_id"] for d in docs}
    return {
        "name": f"BEIR {name} full corpus / test queries (mteb mirror)",
        "source": f"https://huggingface.co/datasets/mteb/{name}/tree/{revision}",
        "source_sha256": {
            p: hashlib.sha256(raw).hexdigest() for p, raw in files.items()
        },
        "limitations": "Graded positive judgments from the BEIR test split; unjudged documents count as not relevant. Titles are passed as metadata, capped at 512 characters.",
        "documents": [
            {
                "id": d["_id"],
                "text": d["text"],
                "metadata": {"title": (d.get("title") or "")[:512]},
            }
            for d in docs
        ],
        "queries": [
            {
                "id": qid,
                "question": queries[qid],
                "relevant": sorted(ids),
                "grades": ids,
                "split": "test",
            }
            for qid, ids in sorted(grades.items())
            if set(ids) <= known
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache" / "rag-pipeline-beir"
    )
    args = parser.parse_args()
    data = prepare(args.dataset, args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False) + "\n")
    print(
        args.dataset,
        len(data["documents"]),
        "documents",
        len(data["queries"]),
        "queries",
    )
