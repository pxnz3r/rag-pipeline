# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [0.3.0] - 2026-10-02

Breaking replacement of the split JSON/Chroma/LightRAG architecture with transactional SQLite FTS5 and optional pinned ONNX dense/reranking models. Removed vulnerable optional stack, graph rebuild/cleanup layers, redundant scripts/tests and obsolete plans. Original stores remain on disk for rollback; reingest source files using the new CLI/API.

Added explicit indexed company/year/jurisdiction filters, TXT/MD/CSV/JSON/JSONL ingestion, bounded PDF worker, original locators/offsets, evidence-only answers, quote-checked optional generation, source-bound decimal calculations and independently labeled row/span evaluations. Models/reranking/generation remain opt-in. See operations, testing and research for measured results and limitations.

## [0.2.0] - 2026-10-02

### Added
- Local `ingest --base-dir` command with bounded PDF chunks and automatic extraction-setting invalidation.
- Structured `QueryResult` with document/page citations and explicit failure states.
- Retrieval quality metrics (recall, precision, MRR, NDCG) for judged query sets.
- Package graph synchronization with isolated generations and atomic publication.
- Real persistent Chroma regression tests and Python 3.10/3.12 CI.
- Repeated benchmarks with median/p95 latency, correctness assertions, and a mutable backend simulator.

### Fixed
- Failed writes and malformed checkpoints no longer silently become successful/empty state.
- Unique temporary files, portable manifest locks, and validated manifest names.
- Removed/empty documents are pruned; stale vectors are deleted after replacement succeeds.
- Cleanup snapshots IDs before deletion to avoid offset-pagination skips.
- Index fingerprints include extraction versions, enriched text, and embedding signatures.
- Enrichment failures remain retryable; empty BM25 corpora no longer crash.
- Independent retrieval fallback, deterministic top-k ties, finite-score handling, async thread offloading, bounded prompts, and redacted generation errors.
- Installed-wheel smoke/ingestion commands no longer depend on repository scripts.

### Changed
- Dependabot version-update PRs disabled for pip and Actions; CI remains enabled with read-only permissions and pinned actions.
- Notebook dependencies moved to a modern optional extra; remote shell installation removed.
- See `docs/AUDIT-2026-10-02.md` for evidence, migration requirements, and remaining upstream advisories.

## [0.1.0] - 2026-03-10

### Added
- Modular `src/rag_pipeline` package extraction from notebook logic.
- CLI entrypoint (`rag-pipeline`) for test, validate, audit, smoke, and benchmark.
- Offline and env-gated live smoke framework.
- Notebook validation and pattern-audit scripts.
- Benchmark harness with memory and latency output.
- CI workflow with tests, notebook checks, and smoke checks.
- Integration tests verifying notebook wrapper delegation.
- Contributor/community docs and GitHub templates.

### Changed
- `Python3finale.ipynb` converted to thin orchestration wrappers over package logic.
- Documentation expanded to architecture, operations, testing, and roadmap.

### Removed
- Obsolete remediation backup assets and redundant helper code paths.
