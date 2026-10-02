# MEMORIES

## Current State
- Public repository: `https://github.com/pxnz3r/rag-pipeline`
- Notebook entrypoint: `Python3finale.ipynb`
- Reusable package: `src/rag_pipeline`
- Tests: `tests/`
- Notebook checks: `scripts/validate_notebook.py`, `scripts/audit_notebook_patterns.py`
- Benchmark harness: `scripts/benchmark_pipeline.py`
- Smoke harness: `scripts/smoke_e2e.py`, `src/rag_pipeline/smoke.py`
- CLI entrypoint: `src/rag_pipeline/cli.py` (`rag-pipeline`)
- Community health files and templates: issue templates, PR template, code owners, dependabot, release config
- CI workflow: `.github/workflows/ci.yml`
- Additional package modules: `models.py`, `storage.py`, `processing.py`, `enrichment.py`, `bm25_utils.py`, `chroma_pipeline.py`, `evaluation.py`, `text.py`
- Query/eval orchestration module: `query_engine.py`
- Removed obsolete remediation artifacts: `Python3finale.remediation_backup.ipynb`, `_remediate_pipeline.py`

## Decisions
- Keep behavior-compatible notebook flow and add package-backed helpers incrementally.
- Enforce notebook quality with CI checks.
- Update pattern-audit rules to support either legacy inline patterns or package-backed imports.
- Prefer package-backed wrappers in notebook bootstrap instead of in-notebook implementations.

## Conventions
- JSON files are read/written with UTF-8.
- Manifest writes are atomic.
- Stale cleanup uses chunked scanning + streaming deletes.

## Gotchas
- Notebook magics (`!pip`, `%`) must be sanitized before AST syntax checks.
- Cell IDs should be present in all notebook cells.

## Open Questions / TODO
- Optional: Add a full live end-to-end smoke run in CI using repository secrets and self-hosted services.

## 2026-10-02 audit update
- Version 0.2.0 adds local ingestion, bounded chunks, structured citations, graph generations, and retrieval quality metrics.
- Checkpoint/manifest corruption now fails visibly; callers must restore or deliberately rebuild state.
- Dependabot version-update PRs disabled; security-update settings are separate and were inaccessible to the session's integration.
- See `docs/AUDIT-2026-10-02.md` for verified findings, coverage, and unresolved upstream advisories.
- Current cleanup scans spool IDs before deleting; do not mutate the backend during pagination.
- Graph generations are retained for recovery; reset creates a new generation.
