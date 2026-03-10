# PLANS

## Purpose / Big Picture
Harden the notebook-based RAG pipeline with reusable modules, tests, benchmarks, and CI.

## Progress
- [x] 2026-03-10 17:35 PKT - Extracted core helpers into `src/rag_pipeline`
- [x] 2026-03-10 17:37 PKT - Added pytest suite for core helpers
- [x] 2026-03-10 17:39 PKT - Added benchmark harness script
- [x] 2026-03-10 17:40 PKT - Added notebook validation and pattern audit scripts
- [x] 2026-03-10 17:46 PKT - Completed notebook wiring to package helpers
- [x] 2026-03-10 17:47 PKT - Ran full validation and captured evidence
- [x] 2026-03-10 18:05 PKT - Migrated additional notebook logic to package modules (processing, enrichment, BM25, Chroma, evaluation data)
- [x] 2026-03-10 18:08 PKT - Updated pattern audit for package-backed architecture
- [x] 2026-03-10 18:20 PKT - Migrated evaluation/retrieval orchestration to package (`query_engine.py`) and wrapped notebook cell 13
- [x] 2026-03-10 18:35 PKT - Removed redundant remediation artifacts and rewrote notebook bootstrap cell to package-backed wrappers
- [x] 2026-03-10 18:55 PKT - Added env-gated smoke path + smoke script + smoke tests
- [x] 2026-03-10 18:58 PKT - Added notebook-wrapper integration tests with mocked delegation
- [x] 2026-03-10 19:00 PKT - Added CLI entrypoint (`rag-pipeline`) + README runbook updates

## Surprises & Discoveries
- Notebook lacked cell IDs, which can become a hard error in future `nbformat` versions.
- Existing pipeline relied heavily on in-notebook definitions without package boundaries.

## Decision Log
- 2026-03-10: Keep notebook as runtime entrypoint while introducing package utilities for gradual migration.

## Outcomes & Retrospective
- Reusable package boundary now exists while notebook remains operational.
- Core regressions are now covered with tests and notebook CI checks.
- Notebook is significantly thinner and delegates major logic to `src/rag_pipeline`.

## Validation and Acceptance
- `pytest` -> 8 passed
- `python scripts/validate_notebook.py Python3finale.ipynb` -> passed
- `python scripts/audit_notebook_patterns.py Python3finale.ipynb` -> passed
- `python scripts/benchmark_pipeline.py --samples 20000` -> baseline generated at `benchmarks/latest.json`
- `pytest` (post-migration) -> 10 passed
- `pytest` (query engine migration) -> 12 passed
- `pytest` (dead-code cleanup) -> 12 passed
- `pytest` (smoke/CLI/wrapper integration) -> 19 passed
- `python scripts/smoke_e2e.py` -> offline smoke passed
- `python -m rag_pipeline.cli smoke` -> offline smoke passed
