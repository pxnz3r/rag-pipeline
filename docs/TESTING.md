# Testing Guide

```bash
python -m pip install --upgrade pip
python -m pip install -e . -r requirements-dev.txt
ruff check src scripts tests
pytest
python scripts/validate_notebook.py Python3finale.ipynb
python scripts/audit_notebook_patterns.py Python3finale.ipynb
rag-pipeline smoke
pip-audit --local
```

Core tests require no credentials, services, or model downloads. Tests cover atomic-write failures, malformed checkpoints, concurrent manifests, PDF extraction/reprocessing, deleted/empty documents, enrichment retries, graph publication failure, real pagination mutation, retrieval fallback, citations, bounded evidence, and event-loop responsiveness.

The real persistent Chroma test is skipped when the optional adapter is absent. Run it without downloading embedding models:

```bash
python -m pip install 'chromadb>=1.5.9,<2'
pytest tests/test_chroma_integration.py
```

CI runs the core suite on Python 3.10 and 3.12 and runs storage integration in a separate job. Core dependency audits apply to the core/developer environment; the optional notebook dependency audit has unresolved upstream advisories documented in the audit report.

## Benchmarks

```bash
rag-pipeline benchmark --samples 20000 --repeats 7 --output benchmarks/latest.json
rag-pipeline benchmark --samples 100000 --repeats 7 --output benchmarks/scale-100k.json
```

The benchmark warms each helper, reports repeated median/p95 timings, compares top-k with a full-sort reference, and checks that cleanup removes every expected orphan. Deletion actually changes the simulator's rows. Earlier non-mutating mocks could hide pagination bugs.

These are synthetic helper benchmarks. They do not measure Chroma embedding throughput, model inference, Groq latency, graph construction cost, or production answer quality. Python allocation measurements exclude the input corpus and native allocations and include simulator copies. Do not compare the new cleanup numbers directly to the old mock's timings.

## Retrieval quality

Use explicit query-to-relevant-chunk judgments:

```python
from rag_pipeline import evaluate_rankings

metrics = evaluate_rankings(
    rankings={"q1": ["chunk-a", "chunk-b"]},
    relevant_ids={"q1": {"chunk-b"}},
    k=5,
)
```

Metrics are macro recall/precision, MRR, and binary NDCG at k. Missing rankings score zero; empty relevance judgments are rejected. These metrics measure retrieval only. The bundled 30-question trading dataset has answer references but no chunk-level relevance labels, so it is not sufficient for a measured production retrieval-quality claim.

## Live checks

`rag-pipeline smoke --live` is a prerequisite check, not a complete live test. Full Groq/Ollama/model/LightRAG/Ragas evaluation must be run in a configured environment with a representative corpus. Do not treat mocked adapter tests or dependency resolution as proof that every live model path works.
