# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, etc.) working in this repository. Humans should start with [CONTRIBUTING.md](CONTRIBUTING.md); this file is the same rules, written for agents.

## Project

`rag-pipeline` is local, evidence-first retrieval. One SQLite transaction holds original text, metadata, FTS5 and optional vectors. Answers return exact evidence with character offsets; optional generation must quote sources verbatim; arithmetic runs as source-bound `Decimal` operations.

- Python 3.11+. Core runtime dependency: `pypdf` only.
- Optional extras: `models` (NumPy, ONNX Runtime, tokenizers, huggingface-hub), `generation` (Groq client).
- CLI entry point: `rag-pipeline` → `rag_pipeline.cli:main`.

## Layout

| Path | Contents |
|---|---|
| `src/rag_pipeline/index.py` | `Index`, `Hit`: SQLite schema, transactional ingest, lexical/dense/hybrid search |
| `src/rag_pipeline/sources.py` | File extraction, metadata sidecars, chunking, heading parsing |
| `src/rag_pipeline/answers.py` | `answer`, `Answer`, `calculate`, `Operand`, quote/unit validation |
| `src/rag_pipeline/reasoning.py`, `expressions.py` | Grounded numerical programs and formulas |
| `src/rag_pipeline/navigation.py` | `Navigation`: budgeted, scoped document browsing |
| `src/rag_pipeline/embeddings.py`, `token_cache.py` | Pinned ONNX encoders, rerankers, ColBERT token store |
| `src/rag_pipeline/configuration.py`, `providers.py` | Explicit model/server/factory configuration |
| `src/rag_pipeline/evaluation.py`, `retrieval_metrics.py` | Judged retrieval evaluation |
| `src/rag_pipeline/cli.py` | Command-line interface |
| `tests/` | pytest suite (real SQLite, files and subprocesses) |
| `scripts/` | Benchmark preparation, runs and comparisons |
| `benchmarks/` | Recorded benchmark results and canaries |
| `docs/` | User and design documentation ([index](docs/README.md)) |
| `Python3finale.ipynb` | Optional notebook interface; calls installed APIs only |

## Commands

```bash
pip install -e . -r requirements-dev.txt
ruff check src scripts tests
ruff format --check src scripts tests
pytest
pip-audit --local
```

Model integration tests (download pinned public weights):

```bash
pip install -e '.[models,generation]'
RAG_TEST_MODELS=1 pytest
```

Run the smallest relevant check first, for example `pytest tests/test_sources.py`, before the full suite.

## Invariants

Do not break these. Add a boundary regression test when you change behavior near one.

- **Provenance:** `Hit.text` and its offsets are always the original source text. Derived data (heading `context`, metadata prefixes) stays in separate fields and is never cited.
- **Transactions:** ingest, publication and migrations are all-or-nothing. A failure rolls back the whole update. A missing source directory fails; it never deletes the corpus.
- **No default model:** no model, vendor or server is chosen implicitly. Retrieval makes no network calls unless the user configured a server.
- **No executed model code:** numerical answers use explicit `Decimal` operations or the restricted expression evaluator. Never `eval` model output, load pickles, or run remote model code.
- **Lean core:** don't add runtime dependencies to the core. Model and generation libraries belong in extras.
- **Package owns policy:** extraction, retrieval and answer logic live in `src/rag_pipeline`. The notebook only calls installed APIs.
- **Index signatures:** if you change extraction, chunking or encoding output, bump the relevant signature in `Index.ingest` so existing indexes rebuild instead of mixing formats.

## Benchmarks

- Record independent relevance judgments before tuning retrieval.
- Report scope, exclusions and failures honestly. Don't claim state-of-the-art or production accuracy.
- Don't edit recorded result files in `benchmarks/` by hand; regenerate them with `scripts/`.
- When a tuning experiment doesn't improve all benchmarks, document it as measured and not adopted.

## Tests

- Prefer a small real-storage test over mocked adapters or wrapper tests.
- Core tests must run offline. Anything that needs model weights goes behind `RAG_TEST_MODELS=1`.

## Git and PRs

- `main` is protected: changes land through a pull request, and the `test-and-audit` check must pass. Force pushes are blocked.
- Branch from `main`, keep PRs focused, and include validation notes. API changes need migration notes.
- Add a line to the `## Unreleased` section of [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
- CI has read-only permissions. Don't add automation that writes branches or opens PRs.
- Don't commit `data/`, `processed_data/`, `.env` files, model weights or SQLite indexes.

## Docs

Update the matching doc when behavior changes: [CLI](docs/CLI.md), [configuration](docs/CONFIGURATION.md), [architecture](docs/ARCHITECTURE.md), [API](docs/API.md), [operations](docs/OPERATIONS.md). Keep the README short; detail goes in `docs/`.
