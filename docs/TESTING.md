# Testing Guide

## Test Layers

- Unit tests for package modules (`tests/`).
- Integration-style checks for notebook wrapper delegation.
- Notebook schema/syntax validation.
- Notebook architectural pattern audit.
- Smoke tests (offline and env-gated live prerequisite checks).

## Commands

```bash
rag-pipeline test
rag-pipeline validate-notebook Python3finale.ipynb
rag-pipeline audit-notebook Python3finale.ipynb
rag-pipeline smoke
rag-pipeline smoke --live
```

## Benchmarking

```bash
rag-pipeline benchmark --samples 20000 --output benchmarks/latest.json
```

Use this to compare runtime characteristics across changes.
