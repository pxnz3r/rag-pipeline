# Tests and measured retrieval

Install `pip install -e . -r requirements-dev.txt`; run `ruff check src scripts tests`, `ruff format --check src scripts tests`, `pytest` and `pip-audit --local`. **40 passed, one model test skipped** in core mode; **41 passed** with `RAG_TEST_MODELS=1` on Python 3.10 and 3.12. The opt-in test loads both actual pinned CPU models. CI audits core and optional model/generation environments and preserves the required `test-and-audit` check. Both local resolved environments report no known advisories; the report is [dependency-audit-0.3.json](dependency-audit-0.3.json), not a proof of source-code security.

The lean suite covers real persistent FTS/vector storage, whole-ingest rollback, WAL reader snapshots, corruption, source collisions/removal, indexed scope and injection-shaped filters, model signatures, invalid vectors/rerank scores, bounded context, actual PDF subprocess parsing, table labels, decimal fidelity, exact citations, forged numerical/currency/scale claims, no-evidence behavior, installed commands and notebook schema/code. The wheel was also installed separately and exercised outside the checkout, including a core environment **without NumPy**. Source distributions include the notebook/canary needed by their tests.

Physical Python + notebook code lines: **3,946 → 2,161 (45% fewer)**. Runtime modules: 20 → 9; test lines: 1,131 → 596; notebook code: 473 → 13. Counts include comments/blank lines, exclude docs/data/generated files and do not rely on minification. Tests of retired adapters were replaced by real boundary cases rather than keeping duplicate mocks.

## Finance retrieval proxy

FinQA official test data: 100 records sampled with seed 42, **2,465 competing text/table rows**, 98 usable questions (two have missing gold rows, listed in the artifact). Uniform row rendering is independent of gold snippets. Gold row IDs come from the original annotations; no per-question company/year filter or tuning is used. This is cross-page row retrieval, **not FinQA execution accuracy or a full benchmark result**.

| Route | Recall@5 | MRR@5 | Median / p95 query ms |
| --- | ---: | ---: | ---: |
| 0.2 BM25 route, raw text | 70.92% | 0.720 | 3.60 / 5.95 |
| 0.2 BM25 with same metadata prefix | 69.73% | 0.697 | 3.86 / 6.37 |
| 0.3 FTS lexical | 69.98% | 0.713 | 1.21 / 2.11 |
| 0.3 hybrid MiniLM + FTS + RRF | 72.28% | 0.755 | 12.68 / 19.03 |
| 0.3 hybrid + cross-encoder | 74.32% | 0.759 | 662.01 / 1,033.51 |

The old baseline is only its BM25 route, not the complete previous dense/graph/reranked/generative system. Lexical alone slightly reduces recall versus raw old BM25; hybrid improves it with additional CPU/index cost. Reranking's small gain does not justify making it mandatory. Index times are about 0.57 seconds lexical and 25.47 seconds hybrid; model loading/downloading is excluded.

## Legal character-span proxy

CUAD official test slice: **10 seed-42 sampled complete contracts, 134 positive questions, 762 chunks**. Original wording/text/expert character spans are preserved. Explicit contract scope matches the dataset's “this contract” question. Document recall is trivial under that scope, so report **macro character coverage** on the union of returned context instead. No-answer clauses are excluded; this is not official CUAD extraction F1 or full LegalBench-RAG.

| Route | Character recall | Character precision | Median / p95 query ms |
| --- | ---: | ---: | ---: |
| FTS lexical | 54.42% | 4.12% | 2.97 / 7.17 |
| Hybrid | **60.07%** | **4.83%** | 12.62 / 20.08 |
| Hybrid + cross-encoder | 53.34% | 3.46% | 1,669.86 / 2,421.58 |

The general MS MARCO reranker **hurts legal retrieval** here. Leave it off for this domain. Low character precision exposes excess surrounding evidence; further legal-specific models/window calibration require a larger independent corpus. Current chunk/context/model defaults were frozen before these labels were evaluated. Rerank batch 8 vs 32 used about **409 vs 831 MiB** peak process RSS on the same 40 legal windows, with identical logits. Profile timing ran under different contention and is not a speed comparison.

## Scope regression and scale

The authored canary has 12 positive heldout queries plus three negative scope/unknown-term cases; recall@3 and negative abstention are 100%. It is deliberately small, author-labeled, and not independent production evidence. It does not establish general unanswerable-question calibration.

A real **100,000-row CSV** ingested into about 35 MiB of SQLite pages in **2.62 s**; unchanged ingest took **6.28 ms**. Broad `revenue 99123` with any-term matching took median/p95 **64.59/66.05 ms**, compared with old in-memory BM25 **38.81/39.45 ms**. The selective all-term lookup returned only the correct account in **2.69/2.72 ms**. These query modes have different result semantics, so do not advertise their latency ratio as an equivalent speedup. SQLite buys durable transactions/bounded memory, not universal lower latency. Dense scans remain linear and were not benchmarked at 100k vectors.

## Reproduce

```bash
pip install -e '.[models]' -r requirements-dev.txt
python scripts/prepare_finqa.py /tmp/finqa-proxy.json
python scripts/prepare_cuad.py /tmp/cuad-proxy.json
rag-pipeline evaluate /tmp/finqa-proxy.json -k 5 --repeats 3
rag-pipeline --dense evaluate /tmp/finqa-proxy.json --mode hybrid -k 5 --repeats 3
rag-pipeline --dense --rerank evaluate /tmp/finqa-proxy.json --mode hybrid -k 5 --repeats 1
rag-pipeline --dense evaluate /tmp/cuad-proxy.json --mode hybrid -k 5 --repeats 3
rag-pipeline evaluate benchmarks/domain-canary.json --split heldout -k 3 --repeats 7
python scripts/benchmark_scale.py --rows 100000 --repeats 7
```

Preparation scripts pin upstream revisions, disclose transformations/exclusions and record SHA-256 provenance. FinQA is MIT (included notice); CUAD is CC BY 4.0 (attributed in its script). Raw external corpora stay outside the repository. Run each route sequentially for comparable timings. One global warm-up is excluded; standard routes have three measured repetitions per question, reranking one due to cost. Shared-worker scheduling affects timing; generated answers, private books and end-to-end hosted service latency were not measured.

[Full metrics, runtime, samples and scope](../benchmarks/results-0.3.json). Previous `latest.json`/`scale-100k.json` artifacts describe historical 0.2 helper benchmarks, not this implementation.
