# Pipeline architecture audit — 2026-10-03

Scope: source extraction through final response, CLI/configuration, persistence, concurrency, resource use and evaluation. This is a source-backed engineering audit, not evidence of market leadership or a guarantee of zero undiscovered defects. Status distinguishes implemented fixes from remaining architectural work.

| Boundary / weakness | Status and evidence |
| --- | --- |
| CLI fixes encoder/reranker choices and silently picks checkpoints | Fixed in 0.5: explicit configuration, arbitrary installed factories, generic ONNX/server adapters; no runtime model default |
| Generation couples CLI to Groq and default Llama | Fixed: independently configured chat endpoint/factory; opt-in generation; Python Groq helper requires model |
| Chat endpoint requires vendor-specific parameters | Fixed in 0.6: explicit copied request options, configurable token-limit field and optional omission of temperature; reserved fields cannot be overridden |
| Server adapter assumes Qwen query formatting | Fixed: explicit query/passage templates with signature identity and plain defaults |
| Immutable HTTP model versions must be Git hashes | Fixed: declared immutable version identifiers accepted; mutable aliases rejected; no server-attestation claim |
| Runtime cannot choose ONNX pooling/output/device | Fixed: explicit pooling, dimensions, graph output and execution providers; unavailable device fails |
| Model identity alone cannot establish vector compatibility | Fixed boundary: signatures include revision, pooling/format settings; mismatches fail; reindex required |
| Hybrid mode silently falls back to lexical | Fixed: lexical default; explicit dense/hybrid requires encoder |
| Config misspellings/duplicate keys are ignored | Fixed: unknown sections/search/answer settings, duplicates and nonfinite values rejected; provider constructors reject unsupported options |
| Evaluation ignores configured candidate/evidence budgets | Fixed: search options passed to warm-up and measured queries and recorded in results |
| Live ingest writer lock spans parsing/model inference | Fixed: changed-document SQLite stage, bounded batches, optimistic concurrency validation and atomic publication; no complete-index clone |
| Token preparation holds writer lock during model inference | Fixed: on-disk token stage from a read snapshot, atomic validated publication; checkpoint/hash checks retained |
| Concurrent preparation can overwrite committed updates | Fixed: generation/data-version/connection-write checks fail on conflict; caller retries explicitly |
| Contextual ingestion shadows metadata with last heading | Fixed: separate heading title variable; extraction version changed; regression checks metadata representation |
| Retrieval issues one SQL metadata query per candidate | Fixed: fetch candidate metadata in bounded SQL batches |
| Token cache restricts models to 512 tokens/4096 dimensions | Fixed: representation geometry accepted within explicit per-vector byte budget; malformed arrays rejected |
| Raw source cap allows normalized extraction amplification | Fixed in 0.6: cumulative UTF-8 budget for repeated CSV labels/JSON indentation and streamed bounded JSON serialization |
| Numerical generation invents operands or computes inaccurately | Fixed boundary in 0.6: complete original numeric tokens, source revisions, bounded step/formula compiler and Decimal execution. Quantity/period/operation selection remains a model error source |
| Source and vector/FTS updates can partially commit | Existing whole-publication transaction and FTS triggers retained; rollback/deletion/migration tests |
| Cache can survive content/model changes incorrectly | Existing body hashes, model signatures, source deletion cascade and read-version invalidation retained |
| Scope leaks through planner-selected IDs | Existing fixed-scope checks across tools/reads; adversarial scope tests retained |
| Generated citations/numbers can be fabricated | Existing exact quote/ID/numeric/unit checks retained; configured generator uses same validation |
| Model responses can be reordered/corrupt/oversized | Validated indexes, finite values, required score coverage and byte limits; local HTTP contract tests |
| Loading heavyweight adapters for status/lexical commands | Lazy role construction retained; no provider is required for core lexical operation |
| ANN scalability | **Remaining:** exact dense scans are O(N×dimension). A bounded warm matrix improves repeated scans, not asymptotic scaling. No million-chunk ANN claim |
| Single persistent-store writer | **Remaining:** final SQLite commits serialize. Staging shortens lock duration; large publication still incurs FTS/vector write cost |
| Query inference while holding a read snapshot | **Remaining:** stable evidence snapshot can retain WAL during slow inference; separate serving workers/connection lifetimes need workload profiling |
| Serving concurrency and scheduling | **Remaining:** one Index connection per owning thread; no async server, queue/backpressure pool or multi-host scheduling implementation |
| Model lifecycle | **Remaining:** CLI launches adapters per invocation; persistent Python application can reuse them. No warm daemon/model eviction scheduler |
| Structured table understanding | **Remaining:** row/header rendering preserves text, not nested headers, merged cells, table relationships or full numerical program induction |
| PDF/image/equation parsing | **Remaining:** no OCR/layout/VLM fidelity guarantees; visual retrieval benchmark not reproduced |
| Chunk selection loses global relationships | **Remaining:** navigation/source ordering available, but explicit complete-context reader and size-based route available; no trained router or graph/multi-hop index |
| Agent planning quality | **Remaining:** bounded tools/callbacks are infrastructure; no trained GRASP/DeepRAG policy or independently evaluated planner |
| Answer semantic correctness | **Remaining:** exact quotations prove text exists, not entailment, correct period, completeness or clinical validity |
| Abstention/calibration | **Remaining:** cosine threshold and no-evidence handling do not establish calibrated unanswerability across models/domains |
| Context budget | Fixed in 0.6: operator-selected evidence/navigation budgets and large reads have no arbitrary context ceiling; read materialization remains bounded by remaining budget. **Remaining:** tokenizer-aware allocation; tokens depend on the configured generator |
| Scope as authorization | **Remaining:** metadata filters are retrieval scope, not tenant security; private library is not a public multitenant service |
| Conversation context and freshness | **Remaining:** caller supplies scope/question; no evaluated conversational memory, trading feeds, validity intervals or contradictory-version reconciliation |
| Source-code semantics | **Remaining:** identifier preservation/text ingestion, not AST/dependency/call-graph retrieval |
| Benchmark competitiveness | **Remaining:** small controlled real-generation QA comparison added; no full official test or matched leading-system comparison establishing top-ten/SOTA |
| Observability/cost accounting | **Remaining:** query timings/tool traces exist, but complete serving-stage percentiles, model-token costs and distributed tracing are absent |
| Cross-domain generalization | **Remaining:** finance/legal proxies and SciFact retrieval; no independent medical QA/accounting execution/trading/code suite |

## Verification and architectural decisions

Inference staging is tested against an actual second SQLite writer during encoding, both rollback and committed-conflict cases. Old corpus state survives preparation failure; peer commits survive conflict rejection. Staged heading parents/ranges remap together with source sections; FK cascade and existing tree/scope tests continue to exercise publication. Token-cache checkpoint replacement remains atomic and reconstructible. Temporary stages contain sensitive extracted text/vectors and use private temporary directories; no remote queue/cache is introduced.

Configured embedding, reranking and chat adapters are exercised through a real local HTTP server and installed CLI ingestion/search/generation, including reversed response ordering and forged citations. This is an end-to-end **contract** test, not answer-quality evidence. Generic ONNX encoder/reranker output is compared with actual pinned reference checkpoints. No synthetic protocol response is reported as a model benchmark.

The independent 0.4 retrieval comparisons remain historical evidence; 0.5 does not claim new accuracy gains merely from configurability or staging. A better deployment needs a chosen corpus scale, concurrency target and end-to-end evaluation protocol. ANN, multimodal parsing and trained routing require measured implementations with the same evidence/version contracts; adding an unverified side index or model policy would reintroduce split-state and evaluation weaknesses.
