# Validation — unreleased (0.8): FinanceBench, parsing, tiers

Same machine as below (4 vCPU Sapphire Rapids, 15 GiB, no GPU). Outputs in [benchmarks/answers-0.8](../benchmarks/answers-0.8).

## FinanceBench evidence pages

All 150 open-source FinanceBench questions over their 84 source filings (`scripts/prepare_financebench.py`, `scripts/benchmark_financebench.py`), PDFium extraction, 57,357 chunks. A question counts when a returned window comes from a gold evidence page; k=10 windows.

| Preset | Recall@10, all 84 filings | Recall@10, the question's filing | Median query |
| --- | ---: | ---: | ---: |
| `lexical` | 20.7% | 50.0% | 71 ms |
| `fast` | 44.7% | 68.0% | 98 ms |
| `hybrid` | 54.7% | 73.3% | 105 ms |
| `rerank` | 64.7% | 74.7% | 3.2 s |

Before the parser change, 41 of the 84 filings failed to ingest at all.

## FinanceBench answers

The first 10 questions, `hybrid` preset, each question scoped to its filing, Qwen3.5-4B Q4_K_M on llama-server (4 CPU threads, 16k context, no reasoning), `scripts/benchmark_answers.py`. Correct means the cited answer's number matches the gold answer within 1% (`numeric_match`); a rejected or abstained answer counts as wrong.

| Route | Correct | Cited | Rejected by validation | Median seconds |
| --- | ---: | ---: | ---: | ---: |
| Single shot, validator before tuning | 0 | 0 | 10 | not recorded |
| Single shot (`ask`) | 4 | 6 | 4 | 178 |
| Agent (`ask --agent`, 6 steps) | 4 | 5 | 4, plus 1 abstained | 326 |

Both routes answered the same four questions correctly. The agent cost 1.8x the time without a gain at this model size. Of the six questions both got wrong:

- Two were cited but wrong (04672 single-shot, 01865 both): the model chose the wrong figure in the right table.
- The agent's 04672 answer cited a net sales total as net PP&E, and the number/unit check rejected it, which was correct.
- The other four were rejected or abstained on both routes: 00499 and 01226 had unverifiable citations, and 02987 and 07966 had unsupported numbers or no answer.

Ten questions is a small sample: one question is 10 points. These rows show the validator now accepts correctly quoted answers, not FinanceBench accuracy. Rows: `financebench-qwen3.5-4b-single.jsonl`, `financebench-qwen3.5-4b-agent.jsonl`, and `financebench-qwen3.5-4b-single-v1-validator.jsonl` for the old validator.

## Postgres store

BEIR NDCG@10, lexical only, same corpora as the SQLite rows below:

| Store and ranking | SciFact | NFCorpus | Median query |
| --- | ---: | ---: | ---: |
| SQLite FTS5 BM25 | 0.6815 | 0.3240 | 7 ms / 1 ms |
| Postgres `ts_rank_cd` (first version) | 0.4299 | 0.2522 | 86 ms / 16 ms |
| Postgres BM25 over postings (shipped) | 0.6867 | 0.3300 | 32 ms / 6 ms |

Row-level security is tested against a real Postgres 16 with pgvector 0.6: a raw connection scoped to one tenant, or to none, reads none of another tenant's rows.

## EmbeddingGemma-300M (int8)

Hybrid, relative-score fusion, against the `hybrid` preset's arctic-embed-m at weight 0.6:

| Dense weight | SciFact | NFCorpus | FinQA R@5 | CUAD char R | FiQA |
| --- | ---: | ---: | ---: | ---: | ---: |
| arctic-embed-m, 0.6 (`hybrid`) | 0.7443 | 0.3664 | 76.11% | 63.72% | 0.3947 |
| EmbeddingGemma, 0.5 | 0.7790 | 0.3843 | 76.62% | 65.32% | not measured |
| EmbeddingGemma, 0.6 | 0.7921 | 0.3932 | 76.11% | 63.03% | not measured |

At weight 0.5 it beats `hybrid` on all four measured sets. FiQA (82,740 chunks) did not finish indexing inside the 2-hour job limit: EmbeddingGemma encodes under 12 chunks/s here against arctic-embed-m's 27, and queries took about 240 ms against 70 ms. The preset stays arctic-embed-m until FiQA is measured.

## PDF extraction

Over the 84 filings, measured with `scripts/compare_pdf_extraction.py` (gold evidence numbers found on the gold page):

| Engine | Failed filings | CPU seconds | Extracted text | Gold numbers found |
| --- | ---: | ---: | ---: | ---: |
| pypdf layout (before) | 41, as shipped | 752 | 114.4 M chars | 99.19% (after decrypting) |
| pypdf plain | 0 after decrypting | 612 | 40.7 M chars | 99.61% |
| PDFium (now) | 0 | 48 | 40.0 M chars | 99.88% |

Docling (`--parser docling`) turned the 160-page 3M 2018 10-K into Markdown tables with rows and columns intact, at about 0.2 pages/s on this contended machine against about 100 pages/s for PDFium. A full FinanceBench comparison would take about 17 hours here, so its retrieval effect is not measured yet.

# Validation — unreleased (0.7)

Machine: 4 vCPU Intel Xeon (Sapphire Rapids, 2.1 GHz, AVX-512 VNNI), 15 GiB RAM, no GPU; Python 3.13, ONNX Runtime 1.29.0, NumPy 2.5.3. Every number below comes from `scripts/benchmark_beir.py` or `rag-pipeline evaluate`; raw outputs are in [benchmarks/beir-0.7](../benchmarks/beir-0.7) and [benchmarks/domain-0.7](../benchmarks/domain-0.7).

## Presets against the previous best route

BEIR: full corpora and test queries from the pinned MTEB mirrors (`scripts/prepare_beir.py`), document ranking (`distinct_documents`, k=10), 40 candidates per channel, NDCG@10 with graded gains. FinQA and CUAD: the same proxies as 0.4 (k=5 evidence windows; FinQA row recall@5, CUAD character recall within the contract scope). Query time is the FiQA median (82,740 chunks).

| Route | SciFact | NFCorpus | FiQA | FinQA R@5 | CUAD char R | FiQA ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 (`lexical`) | 0.6815 | 0.3240 | 0.2394 | 69.98% | 55.99% | 44 |
| MiniLM + BM25, RRF (previous best) | 0.7183 | 0.3495 | 0.3593 | 72.28% | 63.62% | 72 |
| bge-small int8 + BM25, relative 0.6 (`fast`) | 0.7388 | 0.3619 | 0.3946 | 75.85% | 58.58% | 56 |
| arctic-embed-m int8 + BM25, relative 0.6 (`hybrid`) | **0.7443** | **0.3664** | **0.3947** | **76.11%** | **63.72%** | 70 |
| `hybrid` + gte-reranker, top 20 (`rerank`) | **0.7559** | **0.3737** | **0.4415** | **79.42%** | **64.64%** | 3,165 |

`hybrid` is the only first-stage route that beat the previous best on all five; its CUAD gain (+0.1 point) is within noise, so read it as "no legal regression". `fast` wins BEIR and FinQA but loses 5 points of CUAD clause recall. `rerank` improves `hybrid` on all five, most on FiQA (+4.7 NDCG points) and FinQA (+3.3 recall points), at about 3 s per query on 4 cores. The MiniLM row reproduces the recorded 0.4 values on SciFact, FinQA and CUAD exactly.

## What was measured and not adopted

- **Encoders** (hybrid, RRF, SciFact / NFCorpus NDCG@10): bge-small fp32 0.7306 / 0.3627, arctic-embed-s 0.7349 / 0.3531, gte-modernbert int8 0.7408 / 0.3564, nomic-embed-v1.5 int8 0.7269 / 0.3552. arctic-embed-s int8 had the best CUAD recall (65.14%) but fell below the old route on NFCorpus (0.3447). arctic-embed-m-v2 and EmbeddingGemma runs hit the 2-hour job limit and were not measured.
- **Fusion**: relative-score fusion beat RRF at dense weight 0.6 on SciFact and FiQA (+4.1 points on FiQA for bge-small, where BM25 is weak) and tied on NFCorpus (0.3619 vs 0.3627). Weights 0.3 to 0.8 are in `fusion.jsonl` and `domain-0.7/sweep.jsonl`. Library defaults keep unweighted RRF; the presets carry the tuned values.
- **Rerankers** over the bge-small hybrid route, top 20 (SciFact / NFCorpus): jina-reranker-v1-turbo fp32 0.7193 / 0.3565, mxbai-rerank-xsmall fp32 0.7069 / 0.3575, ms-marco MiniLM 0.6806 / 0.3619, mxbai-rerank-base int8 0.6653 / 0.3392. All lowered NDCG. gte-reranker's published int8 export lost most of its gain, and local per-channel int8 (all ops: 0.7159; MatMul only: 0.7341) fell below no reranking, so it runs fp32. At top 10 it adds less (0.7555 / 0.3666) for 1.9 s.
- **ColBERT-small with cached tokens** on the `fast` route: 0.7397 / 0.3716 at 40 ms per query, but the token cache took 428 s and about 600 MB for SciFact alone. It stays an explicit option.
- **Lexical query shape**: running FTS5 ranking before the joins gave identical results at the same speed (43.5 vs 42.4 ms); BM25 scoring itself is the cost.

## Speed

| Change | Before | After |
| --- | ---: | ---: |
| SciFact ingest, bge-small (fp32, 32-chunk window, 4-thread session → int8, 256-chunk window, concurrent 1-thread sessions) | 660 s | 118 s |
| Encoder throughput, bge-small int8, 384 chunks (one 4-thread session → four 1-thread sessions) | 61 chunks/s | 81 chunks/s |
| Query encoding, ArcticM (1-thread → 4-thread non-spinning session) | 15.5 ms | 8.5 ms |
| FiQA dense scan, 82,740 × 768, cached (einsum → BLAS then exact rescoring) | 32 ms | 6.3 ms |
| FiQA dense query over the old 64 MiB default (streamed → 512 MiB cached) | 356 ms | 6.3 ms |
| FiQA hybrid query after query encoding (spinning → non-spinning threads) | 87 ms | 56 ms |
| 48 PDFs × 20 pages, lexical ingest (`scripts/benchmark_pdf_ingest.py`, serial → parallel parsing) | 10.9 s | 2.8 s |

Cached and streamed dense queries returned identical ids and scores on 40 FiQA queries. The core test suite and the `RAG_TEST_MODELS=1` model tests pass on Python 3.13.

## Reproduce

```bash
pip install -e '.[models]'
python scripts/prepare_beir.py fiqa /tmp/beir/fiqa.json
python scripts/benchmark_beir.py /tmp/beir/fiqa.json --work /tmp/beir/work --embedder arctic-m-v1.5-q \
  --mode hybrid --option fusion='"relative"' --option dense_weight=0.6 --reranker gte-reranker --rerank-candidates 20
python scripts/prepare_finqa.py /tmp/finqa.json
rag-pipeline --preset rerank evaluate /tmp/finqa.json -k 5 --repeats 1
```

# Validation — 0.6.1

142 tests pass with actual pinned model assets on Python 3.10 and 3.12. Ordinary dense/hybrid encoding allows concurrent publication and complete WAL truncation; search returns the latest scoped evidence, rejects incompatible model replacement and preserves explicit caller-owned snapshots. On the static 98-question FinQA row proxy, complete Hit dictionaries (including scores, context and source revisions) are identical to merged 0.6; historical 0.5 rankings also match. This is a concurrency fix, not an answer-accuracy gain. [Regression artifact](../benchmarks/query-snapshot-0.6.1.json).

## Historical validation — 0.6

137 tests pass with actual pinned model assets on Python 3.10 and 3.12; both resolved dependency audits report no known vulnerabilities. The installed core-only wheel runs lexical retrieval, grounded formulas, large configured contexts, approved conversions and a 360-month mortgage calculation without NumPy. Numerical programs retain original source revisions and traces; execution validity is distinct from question/period/unit correctness.

The fresh FinQA numeric subset contains 24 report-page-disjoint cases: complete scoped context answers 13/24 correctly (54.2%, Wilson 95% 35.1–72.1); retrieved windows answer 9/24 (37.5%, 21.2–57.3). The paired route difference is 16.7 percentage points in favor of context, with a bootstrap interval spanning zero. This is a quantized 4B, nonthinking, zero-shot CPU profile with supplied report-page scope, not an official full-test or competitive/SOTA score.

A string-only answer estimate/control is not a valid arithmetic baseline: an independent authored USD 125 probe returned brace text with string-or-null output, but 125 when JSON numbers were allowed. The corrected numeric-compatible answer-only control is retained separately and achieves 0/24 exact matches on each route. It is an exploratory, weak control and does not establish broad superiority; stronger few-shot/CoT and frontier-model comparators remain necessary. All failures and the earlier aborted trial are retained. [Raw calls, manifests, intervals, format probe and reproduction](../benchmarks/qa-0.6/README.md). Release replay verifies all 48 primary predictions/statuses and evidence packing unchanged.

Preprocessing on a synthetic unbroken numeric line produces identical catalogs and annotated output: 10,000 numbers 107.54→46.04 ms; 50,000 numbers 2076.15→246.61 ms (8.42×). This isolates quote lookup/annotation construction with inference stopped, not end-to-end latency or accuracy. On the 32-view QA corpus, median work outside generation requests was 12.1 ms retrieval and 3.4 ms context; generation requests took tens of seconds on the 2-CPU quota. [Measurement artifact](../benchmarks/qa-0.6/preprocessing-scale.json).

## Historical validation — 0.5

104 tests pass with actual model assets on Python 3.10/3.12. New tests exercise configuration through local HTTP ingestion/retrieval/reranking/generation and CLI, generic ONNX against reference graph outputs, and real peer writers during staged inference. These are runtime contract checks, not new answer-quality scores. The full 98-question FinQA hybrid replay produces exactly the same per-question rankings as 0.4; [regression artifact](../benchmarks/results-0.5.json). [Architecture audit](ARCHITECTURE-AUDIT.md) distinguishes fixed boundaries from remaining work.

## Historical measured retrieval — 0.4

**93 tests pass with actual pinned models on both Python 3.10 and 3.12.** Core mode skips the one model integration test, which exercises MiniLM, E5, CrossEncoder, ColBERT and ModernColBERT. Both resolved environments have no known dependency advisories. Boundary tests cover atomic migrations/rollback, WAL readers, corrupt caches, scope enforcement, bounded navigation/planner calls, embedding HTTP responses, exact citations/arithmetic and number/unit swaps. The wheel is exercised outside the checkout without NumPy. These checks do not prove every defect absent. Research features add code; the historical 0.3 line-count reduction is not a 0.4 reduction claim.

## Fixed independent comparisons

Reference: main commit `0545897d3b0871046018e7eb86d7240f58d04ed6`. Same prepared corpora and labels, 40 candidate chunks, five returned windows, three timed repetitions after one global warm-up. Window-overlap suppression changes surviving windows, not labels. Some model indexing/queries shared a worker: observed timings are **not matched universal speedups**. CPU quota is two equivalents, approximately 9.7 GiB RAM, no GPU.

FinQA: 100 seed-42 sampled test records, 2,465 competing text/table rows, 98 usable queries; two missing-gold cases disclosed in the artifact. No company/year scope supplied. **Row retrieval, not FinQA arithmetic execution or answer accuracy.**

| Route | Recall@5 | MRR@5 | Median / p95 query ms |
| --- | ---: | ---: | ---: |
| 0.3 lexical | 69.98% | 0.713 | 1.10 / 1.83 |
| 0.4 lexical | 69.98% | 0.713 | 1.05 / 1.79 |
| 0.4 MiniLM hybrid | 72.28% | 0.755 | 6.59 / 9.16 |
| 0.4 E5 hybrid | 72.79% | 0.767 | 5.80 / 7.93 |
| Lexical + cached ColBERT | 76.02% | 0.760 | 20.68 / 58.67 |
| Hybrid + cached ColBERT | 75.51% | 0.758 | 28.52 / 67.58 |
| Lexical + cached ModernColBERT | 77.30% | 0.748 | 111.09 / 199.35 |
| Hybrid + cached ModernColBERT | 77.04% | 0.740 | 191.43 / 297.80 |
| Qwen3-Embedding-0.6B Q8_0 hybrid | 76.19% | 0.779 | 583.89 / 766.85 |

Qwen uses official pinned GGUF through pinned llama.cpp, last-token pooling, instruction-prefixed queries, one CPU inference thread. Indexing took 2,648 seconds under contention. It is a contemporary baseline, not an automatic CPU default. ModernColBERT trades higher recall for lower MRR and more latency than ColBERT-small. All remain explicit options.

Paired bootstrap, 10,000 question resamples, seed 20261003: lexical+ColBERT versus lexical improves recall **6.04 percentage points**, percentile 95% CI **[0.26, 12.41]**. Versus MiniLM hybrid: **3.74 points**, CI **[-0.85, 8.67]**. The latter does not establish a reliable gain on this small slice; no multiplicity-corrected significance or SOTA claim.

CUAD: 10 seed-42 complete contracts, 134 positive questions, 762 chunks. Contract scope follows the original task. Document recall is trivial; measure macro character coverage/precision. Negative clauses excluded; **not official CUAD extraction F1 or full LegalBench-RAG**.

| Route | Character recall | Character precision |
| --- | ---: | ---: |
| 0.3 lexical | 54.42% | 4.12% |
| 0.4 lexical | 55.99% | 3.59% |
| 0.3 MiniLM hybrid | 60.07% | 4.83% |
| 0.4 MiniLM hybrid | 63.62% | 4.19% |
| 0.4 E5 hybrid | 59.73% | 4.29% |
| Lexical + cached ColBERT | 52.87% | 3.27% |
| Hybrid + cached ColBERT | 53.26% | 3.23% |

Diverse windows improve coverage but reduce precision. **ColBERT and E5 regress legal coverage here.** Retain MiniLM hybrid; general ranking scores do not establish legal completeness. The previous MS MARCO cross-encoder also regressed this slice.

SciFact: all 5,183 abstracts and 300 test queries, binary positive judgments. Biomedical retrieval is not clinical answer accuracy or support/refutation. Five-window recall: lexical **73.37% → 75.21%**, MiniLM hybrid **75.72% → 78.92%**, hybrid+ColBERT **79.43%**. Late interaction mainly improves ranking.

## Separate document-ranking protocol

For BEIR-style comparison, explicit `--distinct-documents -k 10` returns ten different source documents. This is a separate protocol, not replacement of five-window results; still a fixed 40-chunk pool, not full PLAID retrieval.

| SciFact route | Document recall@10 | NDCG@10 |
| --- | ---: | ---: |
| Lexical | 80.73% | 0.6815 |
| MiniLM hybrid | 85.26% | 0.7183 |
| Lexical + ColBERT | 83.84% | 0.7284 |
| Hybrid + ColBERT | 85.12% | 0.7402 |

No SOTA claim: model cards use different index/retrieval protocols. Forty-chunk candidate gold coverage is 87.96% lexical / 92.13% hybrid on SciFact, 84.78% / 86.90% on FinQA. Reranking cannot recover missing candidates.

## Context, caching and scale diagnostics

Company/year from the original FinQA input report raises lexical recall to 79.34%, hybrid to 80.36%. This **different known-input-scope protocol** is not an unscoped model gain. Never derive filters from gold evidence.

The six-domain authored heading fixture improves evidence coverage@1 from 33.33% to 100%. It verifies heading preservation, not independent medical/accounting/trading/code quality. Navigation/source ordering/planner callbacks have boundary tests; **no end-to-end agent or long-context answer benchmark was run**. Infrastructure does not reproduce a trained research policy.

Float32 token caching preserves the observed uncached SciFact ranking metrics. ColBERT token payloads alone: FinQA 28.4 MB, SciFact 605.8 MB, CUAD 40.4 MB; ModernColBERT FinQA 42.9 MB. SQLite overhead/model memory are additional; offline encoding costs recorded separately. This is reranking storage, not compressed ANN/PLAID.

Sequential idle-worker synthetic scan ablation: 10,000 vectors, dimension 384, identical generated vectors/query/top-five IDs and scores, one BLAS thread, nine measured repeats. Warm median **21.49 ms in 0.3 → 1.08 ms in 0.4**, fitting the bounded 64 MiB matrix cache. This isolates storage/selection, not inference/accuracy or 100k-vector scaling. Exact search remains linear; filtered/over-budget indexes stream.

Fresh 100,000-row CSV: ingest 2.48 seconds, unchanged ingest 7.05 ms. Broad any-term search median/p95 62.50/62.74 ms; selective all-term 2.69/2.72 ms. Different semantics; not an equivalent speedup or dense-scale benchmark.

## Reproduce

```bash
pip install -e '.[models]' -r requirements-dev.txt
ruff check .
ruff format --check .
pytest
RAG_TEST_MODELS=1 pytest
pip-audit --local
python scripts/prepare_finqa.py /tmp/finqa-proxy.json
python scripts/prepare_cuad.py /tmp/cuad-proxy.json
python scripts/prepare_scifact.py /tmp/scifact.json
rag-pipeline --dense --embedding rag_pipeline.embeddings:MiniLM evaluate /tmp/finqa-proxy.json --mode hybrid -k 5 --repeats 3 --output /tmp/hybrid.json
rag-pipeline --rerank --reranker rag_pipeline.embeddings:ColBERT evaluate /tmp/finqa-proxy.json --mode lexical --cache-reranker -k 5 --repeats 3 --output /tmp/colbert.json
rag-pipeline --dense --embedding rag_pipeline.embeddings:MiniLM --rerank --reranker rag_pipeline.embeddings:ColBERT evaluate /tmp/scifact.json --mode hybrid --cache-reranker --distinct-documents -k 10 --repeats 3
python scripts/compare_results.py /tmp/finqa-proxy.json /tmp/hybrid.json /tmp/colbert.json -k 5
rag-pipeline evaluate benchmarks/context-canary.json --contextual -k 1
OPENBLAS_NUM_THREADS=1 python scripts/benchmark_dense.py --rows 10000 --repeats 9
python scripts/benchmark_scale.py --rows 100000 --repeats 7
```

Run routes sequentially for comparable timing. Preparation scripts pin sources/checksums, disclose exclusions and licenses: FinQA MIT, CUAD CC BY 4.0, SciFact abstract/claim licensing in its script. External corpora remain outside the repository. [Results, provenance, intervals and per-query rankings](../benchmarks/results-0.4.json) preserve regressions/costs. [Research review](RETRIEVAL-LANDSCAPE.md) explains why headline answer scores cannot substitute for retrieval metrics.
