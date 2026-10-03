# Tests and measured retrieval — 0.4

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
rag-pipeline --dense evaluate /tmp/finqa-proxy.json --mode hybrid -k 5 --repeats 3 --output /tmp/hybrid.json
rag-pipeline --rerank --reranker colbert evaluate /tmp/finqa-proxy.json --mode lexical --cache-reranker -k 5 --repeats 3 --output /tmp/colbert.json
rag-pipeline --dense --rerank --reranker colbert evaluate /tmp/scifact.json --mode hybrid --cache-reranker --distinct-documents -k 10 --repeats 3
python scripts/compare_results.py /tmp/finqa-proxy.json /tmp/hybrid.json /tmp/colbert.json -k 5
rag-pipeline evaluate benchmarks/context-canary.json --contextual -k 1
OPENBLAS_NUM_THREADS=1 python scripts/benchmark_dense.py --rows 10000 --repeats 9
python scripts/benchmark_scale.py --rows 100000 --repeats 7
```

Run routes sequentially for comparable timing. Preparation scripts pin sources/checksums, disclose exclusions and licenses: FinQA MIT, CUAD CC BY 4.0, SciFact abstract/claim licensing in its script. External corpora remain outside the repository. [Results, provenance, intervals and per-query rankings](../benchmarks/results-0.4.json) preserve regressions/costs. [Research review](RETRIEVAL-LANDSCAPE.md) explains why headline answer scores cannot substitute for retrieval metrics.
