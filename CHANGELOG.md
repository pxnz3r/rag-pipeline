# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [0.1.0] - 2026-03-10

### Added
- Modular `src/rag_pipeline` package extraction from notebook logic.
- CLI entrypoint (`rag-pipeline`) for test, validate, audit, smoke, and benchmark.
- Offline and env-gated live smoke framework.
- Notebook validation and pattern-audit scripts.
- Benchmark harness with memory and latency output.
- CI workflow with tests, notebook checks, and smoke checks.
- Integration tests verifying notebook wrapper delegation.
- Contributor/community docs and GitHub templates.

### Changed
- `Python3finale.ipynb` converted to thin orchestration wrappers over package logic.
- Documentation expanded to architecture, operations, testing, and roadmap.

### Removed
- Obsolete remediation backup assets and redundant helper code paths.
