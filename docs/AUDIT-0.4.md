# Audit and changes — 2026-10-03

This is an engineering/retrieval audit of 0.3, with boundary regressions and measured comparisons; not a formal registered security scan or proof that every defect is absent.

| Finding | Change / verification |
| --- | --- |
| Passages lose distant heading identity | Opt-in extractive heading hierarchy shared by lexical, dense and reranking representations; original citation text remains separate; six-domain span fixture |
| Seed chunks differ while returned windows repeat | Suppress overlap of actual evidence windows; independent SciFact/CUAD comparison reports coverage and precision separately |
| Query/passage encoders were assumed symmetric | Distinct optional query interface; pinned E5 uses mandatory asymmetric prefixes; signatures prevent incompatible models |
| Stop-word-only queries vanish; language symbols collapse | Preserve `+/#/_` in FTS/query terms, with fallback for standalone stop words; C++/C#/IT/OR and Python/SQL source regression |
| Embedding batches waste CPU on mixed-length padding | Cross-document bounded queue, length grouping and CPU minibatches; restore original vector/logit ordering; discarded large-batch pilot recorded separately |
| Dense top-k converts every block loser to Python objects | NumPy block pruning with all boundary ties retained; exact ID/scoping regression and synthetic scan ablation |
| Repeated dense queries rescan/reconstruct every stored vector | Bounded matrix cache with snapshot/write invalidation; scoped queries retain streaming |
| Opening a reader writes schema state and blocks behind long indexing | Write schema state only during initial creation/migration; real concurrent WAL-writer regression |
| Single pooled vectors discard token-specific relevance | Opt-in projected ColBERT/ModernColBERT token scoring; independent ranking comparisons, no universal default switch |
| Token reranking repeats passage inference on every query | Explicit transactional float32 token cache; source hashes/model signatures, deletion cascade, rollback/corruption checks |
| Retrieval interface cannot navigate original sections or revisit evidence | Scoped bounded document/read/search tools and model-directed action loop; shared citation validator; contract tests do not establish agent QA accuracy |
| Tiny built-in encoders constrain model choices | Bounded self-hosted embedding protocol with immutable artifact signatures and asymmetric query instructions; real Qwen3 CPU baseline |
| Number and unit membership separately admit swaps | Bind explicit adjacent number/unit mentions and recognize common measurement/time units; reject dose/currency/interval swap cases |
| Scientific decimals are unusable; precision ignores exponent span | Explicit scientific operands with bounded exponent magnitude; precision accounts for significant exponent range; exact widely separated amounts regression |
| Python JSON accepts NaN/Infinity | Reject non-finite numeric constants while preserving valid lexical numeric spellings |
| Upgrades risk forcing repeated source rebuilds or partially changing storage | Atomic supported schema migration; preserve sources/vectors/generation; actual failing FTS rebuild restores every schema definition; missing synchronization triggers fail closed |
| New indexes in existing shared directories can inherit broad file modes | New POSIX database/WAL files are owner-only; existing permissions remain operator-controlled |

Remaining constraints are concrete: dense scans are linear, filters scope whole documents rather than individual CSV fields, PDF extraction has no OCR/layout guarantee, and raw Python/SQL ingestion is not AST-aware code understanding. Unit checks have a known catalogue and do not infer table-column relationships, full dosing regimens or semantic entailment. Dense similarity thresholds are not domain-calibrated and do not establish unanswerability. Independent clinical QA, accounting execution, trading freshness and full code retrieval benchmarks are not claimed. No private user corpus or hosted answer-generation evaluation was available.

GitHub automatic security-update administration still returns HTTP 403. Version PR creation remains disabled and CI has read-only repository permissions. See [measured validation](TESTING.md), [research](RESEARCH.md) and [upgrade/rollback](OPERATIONS.md).
