# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, etc.) working in this repository. Humans should start with [CONTRIBUTING.md](CONTRIBUTING.md); this file is the same rules, written for agents.

## Project

`rag-pipeline` is evidence-first retrieval and question answering. The goal is the highest answer accuracy with verifiable citations, on finance, legal, scientific and technical documents. Answers return exact evidence with character offsets; generation must quote sources verbatim; arithmetic runs as source-bound `Decimal` operations. The roadmap is [PLANS.md](PLANS.md).

- Python 3.11+.
- Extras today: `models` (NumPy, ONNX Runtime, tokenizers, huggingface-hub), `generation` (Groq client). Add any dependency, extra, model, hosted API, GPU runtime or service the measurements justify.
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
| `src/rag_pipeline/presets.py` | Named measured stacks (`lexical`, `fast`, `hybrid`, `rerank`) as bundled configurations |
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

## Guarantees

These are the product's value, not limits. Add a boundary regression test when you change behavior near one.

- **Provenance:** `Hit.text` and its offsets are always the original source text, so every citation can be checked. Derived data (heading `context`, LLM-written chunk context, metadata prefixes, parsed tables, page images) lives in separate fields, may feed indexing and ranking, and is never cited as source text.
- **Transactions:** ingest, publication and migrations are all-or-nothing. A failure rolls back the whole update. A missing source directory fails; it never deletes the corpus. Sidecar indexes (ANN, token, image) are rebuilt from the committed store and swapped atomically.
- **No executed model output:** numbers come from explicit `Decimal` operations or the restricted expression evaluator. Never `eval` model output or load untrusted pickles.
- **Index signatures:** if you change extraction, chunking, enrichment or encoding output, bump the relevant signature in `Index.ingest` so existing indexes rebuild instead of mixing formats.

## Open choices

Nothing else is off the table. Pick by measurement:

- Any model, size or vendor: local ONNX, GGUF, PyTorch, GPU or Apple execution providers, hosted embedding, reranking and chat APIs. Defaults and presets may select models.
- Network calls when the configured route needs them, with API keys read from the environment.
- Any dependency, storage engine, sidecar index, server or daemon mode.
- LLM work at ingest or query time: contextual chunk descriptions, query rewriting, agentic multi-step retrieval, listwise reranking.

## Benchmarks

- Judge changes on the suite in `docs/TESTING.md`: BEIR SciFact, NFCorpus, FiQA, the FinQA and CUAD proxies, plus answer-level FinanceBench and LegalBench-RAG once added (`PLANS.md` phase 0).
- Adopt a change that wins the suite on balance; record every per-benchmark regression next to it.
- Report scope, exclusions and failures honestly, and claim only what the measurements show.
- Don't edit recorded result files in `benchmarks/` by hand; regenerate them with `scripts/`.

## Tests

- Prefer a small real-storage test over mocked adapters or wrapper tests.
- The default test run stays offline so CI needs no keys. Tests that need model weights go behind `RAG_TEST_MODELS=1`; tests that call hosted APIs go behind `RAG_TEST_HOSTED=1`.

## Git and PRs

- `main` is protected: changes land through a pull request, and the `test-and-audit` check must pass. Force pushes are blocked.
- Branch from `main`, keep PRs focused, and include validation notes. API changes need migration notes.
- Add a line to the `## Unreleased` section of [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
- CI has read-only permissions. Don't add automation that writes branches or opens PRs.
- Don't commit `data/`, `processed_data/`, `.env` files, model weights or SQLite indexes.

## Docs

Update the matching doc when behavior changes: [CLI](docs/CLI.md), [configuration](docs/CONFIGURATION.md), [architecture](docs/ARCHITECTURE.md), [API](docs/API.md), [operations](docs/OPERATIONS.md). Keep the README short; detail goes in `docs/`.
