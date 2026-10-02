# CLI Reference

Installed-package commands:

- `rag-pipeline ingest [--base-dir PATH]`: synchronize local PDFs into an atomic master checkpoint. Offline; no credentials or model downloads. Reports chunk/document counts and exits nonzero on parse, configuration, or persistence failure.
- `rag-pipeline smoke [--live]`: offline retrieval smoke by default; `--live` checks credential presence and Ollama reachability without paid calls.

Developer commands require a source checkout and development dependencies:

- `rag-pipeline test`: run pytest in the current directory. Run from the repository root.
- `rag-pipeline validate-notebook [PATH]`: check notebook schema and Python syntax, with magics sanitized. Defaults to `Python3finale.ipynb`.
- `rag-pipeline audit-notebook [PATH]`: check notebook architecture patterns. This is a structural check, not a complete security audit.
- `rag-pipeline benchmark [--samples N] [--repeats N] [--output PATH]`: repeated helper benchmarks with median/p95 latency and correctness checks. Default samples: 20000, repeats: 5, output: `benchmarks/latest.json`.

For source-directory discovery, script-based developer commands first use the installed editable checkout and otherwise use the current repository root. A wheel installed without a checkout cannot run repository-only scripts.
