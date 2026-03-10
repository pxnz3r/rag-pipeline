# Operations Runbook

## Prerequisites

- Python 3.10+
- Optional live dependencies:
  - `GROQ_API_KEY`
  - Local Ollama service (default `http://localhost:11434`)

## Common Commands

```bash
rag-pipeline test
rag-pipeline validate-notebook Python3finale.ipynb
rag-pipeline audit-notebook Python3finale.ipynb
rag-pipeline smoke
rag-pipeline benchmark --samples 20000 --output benchmarks/latest.json
```

## Live Smoke

```bash
rag-pipeline smoke --live
```

Live smoke verifies env prerequisites and Ollama reachability.

## Failure Triage

- Notebook validation fails:
  - Ensure cell IDs are present.
  - Fix syntax in code cells (magics are sanitized during checks).
- Pattern audit fails:
  - Confirm notebook wrappers delegate to `src/rag_pipeline`.
- Smoke fails:
  - Offline: investigate package-level retrieval wrappers.
  - Live: validate `GROQ_API_KEY` and Ollama availability.
