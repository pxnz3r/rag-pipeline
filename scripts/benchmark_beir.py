"""Sweep retrieval routes over a prepared BEIR set, reusing one index per encoder.

    python scripts/prepare_beir.py scifact /tmp/beir/scifact.json
    python scripts/benchmark_beir.py /tmp/beir/scifact.json --work /tmp/beir/work \
        --embedder bge-small --mode hybrid --reranker none --output results.jsonl

The index is built once per (dataset, encoder, chunking) and reused, so reranker
and fusion sweeps don't re-encode the corpus. Each run appends one JSON line with
document-level NDCG@k (graded gains), recall@k, MRR@k and query latency.
"""

import argparse
import json
import time
from pathlib import Path
from statistics import median

from rag_pipeline import Index
from rag_pipeline.evaluation import _percentile
from rag_pipeline.retrieval_metrics import evaluate_rankings


class Registry(dict):
    """Model names; a "-q" suffix loads the published int8 export of the same commit."""

    def __missing__(self, name):
        base = self[name.removesuffix("-q")]
        if not name.endswith("-q") or not hasattr(base, "options"):
            raise KeyError(name)
        return base.with_weights("onnx/model_quantized.onnx")


class Factory:
    def __init__(self, kind, **options):
        self.kind, self.options = kind, options

    def with_weights(self, weights):
        return Factory(self.kind, **{**self.options, "weights": weights})

    def __call__(self, threads):
        return self.kind(threads=threads, **self.options)


def models():
    from rag_pipeline import embeddings as e

    def onnx(model, revision, **options):
        return Factory(e.ONNXEmbedding, model=model, revision=revision, **options)

    def rerank(model, revision, **options):
        return Factory(e.ONNXReranker, model=model, revision=revision, **options)

    bge_query = "Represent this sentence for searching relevant passages: "
    return Registry(
        {
            "minilm": lambda threads: e.MiniLM(threads=threads),
            "e5-small": lambda threads: e.E5(threads=threads),
            "bge-small": onnx(
                "BAAI/bge-small-en-v1.5",
                "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                dimensions=384,
                length=512,
                pooling="cls",
                query_prefix=bge_query,
                batch_size=4,
            ),
            "bge-small-q8": onnx(
                "Xenova/bge-small-en-v1.5",
                "ea104dacec62c0de699686887e3f920caeb4f3e3",
                weights="onnx/model_quantized.onnx",
                dimensions=384,
                length=512,
                pooling="cls",
                query_prefix=bge_query,
                batch_size=4,
            ),
            "bge-base": onnx(
                "BAAI/bge-base-en-v1.5",
                "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
                dimensions=768,
                length=512,
                pooling="cls",
                query_prefix=bge_query,
                batch_size=4,
            ),
            "arctic-s": onnx(
                "Snowflake/snowflake-arctic-embed-s",
                "e596f507467533e48a2e17c007f0e1dacc837b33",
                dimensions=384,
                length=512,
                pooling="cls",
                query_prefix=bge_query,
                batch_size=4,
            ),
            "arctic-m-v1.5": onnx(
                "Snowflake/snowflake-arctic-embed-m-v1.5",
                "e58a8f756156a1293d763f17e3aae643474e9b8a",
                dimensions=768,
                length=512,
                pooling="cls",
                query_prefix=bge_query,
                batch_size=4,
            ),
            "arctic-m-v2": onnx(
                "Snowflake/snowflake-arctic-embed-m-v2.0",
                "95c2741480856aa9666782eb4afe11959938017f",
                dimensions=768,
                length=512,
                pooling="cls",
                query_prefix="query: ",
                batch_size=4,
            ),
            "nomic-v1.5": onnx(
                "nomic-ai/nomic-embed-text-v1.5",
                "e9b6763023c676ca8431644204f50c2b100d9aab",
                dimensions=768,
                length=512,
                pooling="mean",
                query_prefix="search_query: ",
                passage_prefix="search_document: ",
                batch_size=4,
            ),
            "gte-modernbert": onnx(
                "Alibaba-NLP/gte-modernbert-base",
                "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
                dimensions=768,
                length=512,
                pooling="cls",
                batch_size=4,
            ),
            "embeddinggemma": onnx(
                "onnx-community/embeddinggemma-300m-ONNX",
                "5090578d9565bb06545b4552f76e6bc2c93e4a66",
                dimensions=768,
                length=512,
                pooling="pooled",
                output_index=1,
                query_prefix="task: search result | query: ",
                passage_prefix="title: none | text: ",
                batch_size=4,
            ),
            "ms-marco-minilm": lambda threads: e.CrossEncoder(threads=threads),
            "colbert-small": lambda threads: e.ColBERT(threads=threads),
            "gte-reranker": rerank(
                "Alibaba-NLP/gte-reranker-modernbert-base",
                "f7481e6055501a30fb19d090657df9ec1f79ab2c",
                length=512,
                batch_size=4,
            ),
            "mxbai-xsmall": rerank(
                "mixedbread-ai/mxbai-rerank-xsmall-v1",
                "b5c6e9da73abc3711f593f705371cdbe9e0fe422",
                length=512,
                batch_size=4,
            ),
            "mxbai-base": rerank(
                "mixedbread-ai/mxbai-rerank-base-v1",
                "800f24c113213a187e65bde9db00c15a2bb12738",
                length=512,
                batch_size=4,
            ),
            "bge-reranker-base": rerank(
                "BAAI/bge-reranker-base",
                "2cfc18c9415c912f9d8155881c133215df768a70",
                length=512,
                batch_size=4,
            ),
            "jina-turbo": rerank(
                "jinaai/jina-reranker-v1-turbo-en",
                "b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2",
                length=512,
                batch_size=4,
            ),
        }
    )


def corpus(data, directory):
    """Write each document once; ingest is incremental, so reruns skip parsing."""
    names = {}
    directory.mkdir(parents=True, exist_ok=True)
    for i, doc in enumerate(data["documents"]):
        name = f"{i:08d}.txt"
        names[name] = doc["id"]
        path = directory / name
        if not path.exists():
            path.write_text(doc["text"], encoding="utf-8")
            (directory / (name + ".meta.json")).write_text(
                json.dumps(doc.get("metadata", {}))
            )
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--embedder", default="none")
    parser.add_argument("--reranker", default="none")
    parser.add_argument(
        "--mode", default="lexical", choices=["lexical", "dense", "hybrid"]
    )
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--rerank-candidates", type=int, default=None)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="First N queries only")
    parser.add_argument(
        "--option", action="append", default=[], help="Extra search option key=json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    name = args.dataset.stem
    registry = models()
    embedder = (
        None if args.embedder == "none" else registry[args.embedder](args.threads)
    )
    reranker = (
        None if args.reranker == "none" else registry[args.reranker](args.threads)
    )
    files = corpus(data, args.work / f"{name}-corpus")
    path = args.work / f"{name}-{args.embedder}-{args.size}-{args.overlap}.sqlite"
    options = {k: json.loads(v) for k, v in (o.split("=", 1) for o in args.option)}
    if args.rerank_candidates:
        options["rerank_candidates"] = args.rerank_candidates
    with Index(path, embedder, reranker, vector_cache_bytes=1024**3) as index:
        started = time.perf_counter()
        stats = index.ingest(
            args.work / f"{name}-corpus", size=args.size, overlap=args.overlap
        )
        ingest_seconds = time.perf_counter() - started
        queries = data["queries"][: args.limit or None]
        search = dict(
            mode=args.mode,
            k=args.k,
            candidates=args.candidates,
            distinct_documents=True,
            **options,
        )
        index.search(queries[0]["question"], **search)
        rankings, samples = {}, []
        for query in queries:
            started = time.perf_counter()
            hits = index.search(query["question"], **search)
            samples.append((time.perf_counter() - started) * 1000)
            rankings[query["id"]] = list(dict.fromkeys(files[h.document] for h in hits))
    qrels = {q["id"]: set(q["relevant"]) for q in queries}
    grades = {q["id"]: q.get("grades") for q in queries if q.get("grades")}
    metrics = evaluate_rankings(rankings, qrels, args.k, grades)
    result = {
        "dataset": name,
        "embedder": args.embedder,
        "reranker": args.reranker,
        "mode": args.mode,
        "candidates": args.candidates,
        "options": options,
        "size": args.size,
        "overlap": args.overlap,
        "chunks": stats["chunks"],
        "changed": stats["changed"],
        "ingest_seconds": round(ingest_seconds, 2),
        f"ndcg@{args.k}": round(metrics["ndcg_at_k"], 4),
        f"recall@{args.k}": round(metrics["recall_at_k"], 4),
        f"mrr@{args.k}": round(metrics["mrr_at_k"], 4),
        "queries": metrics["queries"],
        "query_ms_median": round(median(samples), 2),
        "query_ms_p95": round(_percentile(samples, 0.95), 2),
    }
    print(json.dumps(result))
    if args.output:
        with args.output.open("a") as stream:
            stream.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
