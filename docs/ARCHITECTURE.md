# Architecture

## Overview

The system follows a notebook-orchestrator + package-core architecture:
- `Python3finale.ipynb`: interactive orchestration, experiments, and workflows.
- `src/rag_pipeline`: reusable, testable runtime logic.

## Core Modules

- `config.py`: typed settings and env-driven configuration.
- `processing.py`: PDF chunk extraction and master chunk synchronization.
- `enrichment.py`: context enrichment with checkpoint-safe persistence.
- `bm25_utils.py`: tokenization and BM25 index construction.
- `chroma_pipeline.py`: Chroma ingestion, stale removal, manifest sync.
- `query_engine.py`: hybrid retrieval + reranking + evaluation orchestration.
- `manifest.py`: atomic manifest operations.
- `cleanup.py`: chunked orphan scanning and bounded deletion.
- `observability.py`: structured logging.
- `smoke.py`: offline and env-gated live smoke checks.

## Design Principles

- Move logic out of notebook cells into package modules.
- Keep state transitions explicit and durable (atomic writes, manifests).
- Use bounded-memory iterators for cleanup and large scans.
- Guard expensive or external operations with clear checks.
- Prefer deterministic utilities and tested wrappers.
