# RAG Pipeline

[![CI](https://img.shields.io/github/actions/workflow/status/pxnz3r/rag-pipeline/ci.yml?branch=main&label=CI)](https://github.com/pxnz3r/rag-pipeline/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Production-minded RAG project with:
- A notebook orchestrator for interactive workflows.
- A modular Python package for testable core logic.
- CI quality gates, smoke checks, benchmarks, and structured docs.

## Why This Project

`rag-pipeline` is designed for teams that want notebook speed without notebook fragility.  
Core logic is moved into `src/rag_pipeline`, while the notebook remains a thin orchestration layer.

## Core Capabilities

- Hybrid retrieval architecture (dense + BM25 + reranking).
- Manifest-based ingestion safety with atomic writes.
- Chunked stale cleanup with bounded-memory deletion.
- Offline and env-gated live smoke checks.
- Notebook validation and architectural pattern auditing.
- CLI-first developer workflow for testing, auditing, and benchmarking.

## Project Layout

```text
.
|-- Python3finale.ipynb
|-- src/rag_pipeline/
|   |-- cli.py
|   |-- query_engine.py
|   |-- chroma_pipeline.py
|   |-- processing.py
|   |-- enrichment.py
|   |-- bm25_utils.py
|   |-- manifest.py
|   |-- cleanup.py
|   |-- storage.py
|   |-- config.py
|   |-- observability.py
|   `-- ...
|-- tests/
|-- scripts/
|-- docs/
`-- .github/workflows/ci.yml
```

## Quick Start

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## CLI Runbook

```bash
# test suite
rag-pipeline test

# notebook checks
rag-pipeline validate-notebook Python3finale.ipynb
rag-pipeline audit-notebook Python3finale.ipynb

# smoke checks
rag-pipeline smoke
rag-pipeline smoke --live

# benchmark
rag-pipeline benchmark --samples 20000 --output benchmarks/latest.json
```

## Notebook Workflow

Use `Python3finale.ipynb` for interactive iteration.  
Keep logic changes in `src/rag_pipeline` and use notebook wrappers only for orchestration.

## Quality Gates

CI validates:
- tests
- notebook schema/syntax
- notebook architectural patterns
- offline smoke
- CLI smoke

## Configuration

Main runtime knobs can be supplied via environment variables (for example):
- `PIPELINE_VERSION`
- `GROQ_TIMEOUT`
- `RETRIEVAL_TOP_K`
- `RERANK_CANDIDATES`
- `FINAL_TOP_K`
- `MAX_CHECKPOINT_BYTES`

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Operations Runbook](docs/OPERATIONS.md)
- [Testing Guide](docs/TESTING.md)
- [CLI Reference](docs/CLI.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)

## License

This project is licensed under the [MIT License](LICENSE).
