"""Trial a learned sparse channel against an existing BEIR index, before building it in.

    python scripts/trial_sparse.py /tmp/beir/scifact.json \
        --index /tmp/beir/work/scifact-embeddinggemma-q-900-100.sqlite \
        --embedder embeddinggemma-q --output sparse.jsonl

Encodes every indexed chunk's search text with an inference-free sparse encoder
(OpenSearch neural sparse doc-v3: documents through a masked-LM head with
max-pooled log(1+log(1+relu)) weights; queries are their tokens' IDF, so a query
costs a lookup, not a model call), caches the sparse matrix next to the index,
then scores BM25 (the index's own FTS5 query), dense (the index's own vectors)
and sparse per query and evaluates fused combinations at document level, as
benchmark_beir.py does. Needs torch and transformers; nothing is written into
the index.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_beir import corpus, models  # noqa: E402

from rag_pipeline import Index  # noqa: E402
from rag_pipeline.index import query_terms  # noqa: E402
from rag_pipeline.retrieval_metrics import evaluate_rankings  # noqa: E402

MODELS = {
    "doc-v3-distill": (
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
        "babf71f3c48695e2e53a978208e8aba48335e3c0",
    ),
    "doc-v3-gte": (
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
        "1646fef40807937e8e130c66d327a26421c408d5",
    ),
}
# (bm25, sparse, dense) weights; each channel is min-max normalized per query.
COMBINATIONS = {
    "bm25": (1, 0, 0),
    "sparse": (0, 1, 0),
    "dense": (0, 0, 1),
    "bm25+dense 0.6": (0.4, 0, 0.6),
    "sparse+dense 0.5": (0, 0.5, 0.5),
    "sparse+dense 0.6": (0, 0.4, 0.6),
    "bm25+sparse": (0.5, 0.5, 0),
    "all 0.2/0.2/0.6": (0.2, 0.2, 0.6),
    "all 0.3/0.3/0.4": (0.3, 0.3, 0.4),
}


def encode(texts, name, threads, batch=16):
    """CSR matrix of document term weights, rows in text order."""
    import numpy as np
    import scipy.sparse as sp
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    repo, revision = MODELS[name]
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForMaskedLM.from_pretrained(
        repo, revision=revision, trust_remote_code=name == "doc-v3-gte"
    ).eval()
    special = list(tokenizer.all_special_ids)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    rows, cols, vals, positions = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            feature = tokenizer(
                [texts[i] for i in ids],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            logits = model(**feature)[0]
            values = (logits * feature["attention_mask"].unsqueeze(-1)).amax(dim=1)
            values = torch.log1p(torch.log1p(torch.relu(values)))
            values[:, special] = 0
            r, c = torch.nonzero(values, as_tuple=True)
            rows.append((r + start).numpy())
            cols.append(c.numpy())
            vals.append(values[r, c].numpy())
            positions.extend(ids)
    sorted_matrix = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(texts), tokenizer.vocab_size),
        dtype=np.float32,
    )
    inverse = np.empty(len(texts), dtype=np.int64)
    inverse[np.asarray(positions)] = np.arange(len(texts))
    return sorted_matrix[inverse], tokenizer


def query_vectors(questions, name, tokenizer):
    """Inference-free queries: each distinct token weighted by the model's IDF."""
    from huggingface_hub import hf_hub_download

    repo, revision = MODELS[name]
    idf = json.loads(
        Path(hf_hub_download(repo, "idf.json", revision=revision)).read_text()
    )
    weights = {tokenizer.convert_tokens_to_ids(t): w for t, w in idf.items()}
    special = set(tokenizer.all_special_ids)
    vectors = []
    for question in questions:
        ids = set(tokenizer(question, truncation=True, max_length=512)["input_ids"])
        vectors.append({i: weights[i] for i in ids - special if i in weights})
    return vectors


def top(scores, ids, candidates):
    """(chunk id, score) best first, ties on id, positive scores only."""
    import numpy as np

    positive = np.flatnonzero(scores > 0)
    if len(positive) > candidates:
        cutoff = np.partition(scores[positive], -candidates)[-candidates]
        positive = positive[scores[positive] >= cutoff]
    ranked = sorted(((-float(scores[i]), ids[i]) for i in positive))[:candidates]
    return [(cid, -negative) for negative, cid in ranked]


def fuse(rankings, weights):
    scores = {}
    for ranking, weight in zip(rankings, weights):
        if not ranking or not weight:
            continue
        low, high = ranking[-1][1], ranking[0][1]
        for cid, value in ranking:
            relative = (value - low) / (high - low) if high > low else 1
            scores[cid] = scores.get(cid, 0) + weight * relative
    return sorted(scores, key=lambda c: (-scores[c], c))


def main():
    import numpy as np
    import scipy.sparse as sp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embedder", required=True)
    parser.add_argument("--sparse", default="doc-v3-distill", choices=sorted(MODELS))
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    files = corpus(data, args.index.parent / f"{args.dataset.stem}-corpus")
    embedder = models()[args.embedder](args.threads)
    with Index(args.index, embedder, vector_cache_bytes=1024**3) as index:
        chunks = index.db.execute(
            "SELECT c.id,c.search_text,s.document FROM chunks c "
            "JOIN sections s ON s.id=c.section ORDER BY c.rowid"
        ).fetchall()
        ids = [r[0] for r in chunks]
        document = {r[0]: files[r[2]] for r in chunks}
        cache = args.index.with_suffix(f".{args.sparse}.npz")
        started = time.perf_counter()
        if cache.exists():
            from transformers import AutoTokenizer

            repo, revision = MODELS[args.sparse]
            matrix = sp.load_npz(cache).tocsr()
            tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
            if matrix.shape[0] != len(ids):
                raise ValueError(f"{cache} does not match the index; delete it")
            encode_seconds = None
        else:
            texts = [r[1] for r in chunks]
            matrix, tokenizer = encode(texts, args.sparse, args.threads)
            encode_seconds = round(time.perf_counter() - started, 1)
            sp.save_npz(cache, matrix)
        csc = matrix.tocsc()
        queries = data["queries"]
        sparse_queries = query_vectors(
            [q["question"] for q in queries], args.sparse, tokenizer
        )
        clause, filter_args = index._filters(None)
        channels, sparse_ms = {}, []
        for query, vector in zip(queries, sparse_queries):
            terms = query_terms(query["question"])
            expression = " OR ".join('"' + t + '"' for t in terms)
            bm25 = (
                [
                    (r[0], -r[1])
                    for r in index.db.execute(
                        "SELECT c.id,fts.rank FROM fts JOIN chunks c ON c.rowid=fts.rowid "
                        "WHERE fts MATCH ? ORDER BY fts.rank LIMIT ?",
                        [expression, args.candidates],
                    )
                ]
                if terms
                else []
            )
            dense = [
                (cid, -negative)
                for negative, cid in index._dense_ranking(
                    index._vectors([query["question"]], write=False)[0],
                    clause,
                    filter_args,
                    args.candidates,
                    0.3,
                )
            ]
            begun = time.perf_counter()
            tokens = list(vector)
            scores = (
                np.asarray(
                    csc[:, tokens]
                    @ np.asarray([vector[t] for t in tokens], dtype=np.float32)
                ).ravel()
                if tokens
                else np.zeros(len(ids), dtype=np.float32)
            )
            sparse = top(scores, ids, args.candidates)
            sparse_ms.append((time.perf_counter() - begun) * 1000)
            channels[query["id"]] = (bm25, sparse, dense)
    qrels = {q["id"]: set(q["relevant"]) for q in queries}
    grades = {q["id"]: q.get("grades") for q in queries if q.get("grades")}
    nonzero = matrix.getnnz(axis=1)
    for label, weights in COMBINATIONS.items():
        rankings = {
            qid: list(dict.fromkeys(document[c] for c in fuse(rankings, weights)))
            for qid, rankings in channels.items()
        }
        metrics = evaluate_rankings(rankings, qrels, args.k, grades)
        result = {
            "dataset": args.dataset.stem,
            "embedder": args.embedder,
            "sparse": args.sparse,
            "route": label,
            "weights": weights,
            "candidates": args.candidates,
            "chunks": len(ids),
            f"ndcg@{args.k}": round(metrics["ndcg_at_k"], 4),
            f"recall@{args.k}": round(metrics["recall_at_k"], 4),
            "queries": metrics["queries"],
            "sparse_encode_seconds": encode_seconds,
            "sparse_terms_per_chunk": round(float(nonzero.mean()), 1),
            "sparse_query_ms_median": round(float(np.median(sparse_ms)), 2),
        }
        print(json.dumps(result))
        if args.output:
            with args.output.open("a") as stream:
                stream.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
