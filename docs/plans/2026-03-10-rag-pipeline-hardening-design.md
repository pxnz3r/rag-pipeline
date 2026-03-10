# RAG Pipeline Hardening Design (2026-03-10)

## Purpose
Move the notebook from ad-hoc execution toward a reproducible, testable, and auditable project without breaking existing notebook flow.

## Scope
- Extract reusable core helpers into a package (`src/rag_pipeline`).
- Add regression tests for known fragile paths.
- Add benchmark harness for latency/memory baselines.
- Add CI checks for tests + notebook schema/syntax + pattern audits.
- Keep notebook behavior backward compatible.

## Architecture
- `config.py`: typed runtime settings + env overrides.
- `manifest.py`: atomic manifest IO with bulk update + stale-entry removal.
- `retrieval.py`: ranking helpers and small retrieval utilities.
- `cleanup.py`: chunked orphan scanning and streaming deletions.
- `observability.py`: JSON-structured logging primitives.

## Data Flow
Notebook remains entrypoint.
Notebook can progressively import helpers from `rag_pipeline` while preserving local fallbacks.
CI enforces notebook quality gates on every PR.

## Error Handling
- Prefer narrow exceptions on file/JSON operations.
- Recover to safe defaults where possible (`{}` for manifest read failure).
- Preserve atomic writes (`tmp + fsync + replace`).

## Testing
- Unit tests for manifest sync behavior.
- Unit tests for ranking helpers.
- Unit tests for cleanup iterators and streaming deletes.

## Performance
- Benchmark harness tracks top-k selection, chunk grouping, and orphan cleanup timings.
- Include memory peak metrics to catch regressions.

## Risks and Mitigations
- Risk: notebook/package divergence.
  - Mitigation: CI pattern audit and progressive import wiring.
- Risk: local env differences for notebook execution.
  - Mitigation: central settings and explicit env-driven overrides.
