# RAG Pipeline

[![CI](https://img.shields.io/github/actions/workflow/status/pxnz3r/rag-pipeline/ci.yml?branch=main&label=CI)](https://github.com/pxnz3r/rag-pipeline/actions/workflows/ci.yml)

Local, evidence-first retrieval for financial reports, account records, legal contracts, and books. One SQLite transaction synchronizes original text, metadata, full-text search and optional vectors. The core has only one third-party dependency (pypdf); NumPy/inference stay in the models extra. Default answers return exact evidence; optional generation must provide verifiable quotes. Numerical reasoning uses explicit source-bound decimal operations.

```bash
python -m pip install --upgrade pip
pip install -e .
mkdir -p data
# Add PDFs, TXT/Markdown, CSV, JSON or JSONL to data/.
rag-pipeline ingest data
rag-pipeline search "revenue" --filter company=Acme --filter year=2024
rag-pipeline ask "revenue" --filter company=Acme --filter year=2024
```

Add `report.pdf.meta.json` beside `report.pdf`, for example:

```json
{"company":"Acme","year":2024,"kind":"annual_report","jurisdiction":"NY"}
```

Filters apply before lexical and dense scoring. CSV rows repeat column labels; PDFs retain page locators; text evidence includes original character offsets. Changing a file, its metadata or chunking/model configuration invalidates its index. Failed ingestion rolls back the entire update. A missing source directory never silently deletes the corpus.

Optional CPU semantic retrieval and reranking:

```bash
pip install -e '.[models]'
rag-pipeline --dense ingest data
rag-pipeline --dense search "How much did sales increase?" --filter company=Acme
rag-pipeline --dense --rerank search "Which obligations survive termination?" --filter jurisdiction=NY
```

Model assets are pinned ONNX files; no PyTorch, pickle caches, graph database, or remotely executable model code. Dense search scans filtered vectors in bounded batches. Reranking costs more and stays opt-in. First use downloads public weights. No data leaves the machine for retrieval; `ask --generate` explicitly sends evidence to Groq and requires `.[generation]` plus `GROQ_API_KEY`.

```python
from rag_pipeline import Index, answer

with Index("processed_data/index.sqlite") as index:
    result = answer(index, "What was revenue?", filters={"company": "Acme", "year": "2024"})
```

Measured checks include independent FinQA row judgments, CUAD legal character spans, an authored regression canary, and 100,000 account rows. See [results and reproduction](docs/TESTING.md) and [research](docs/RESEARCH.md). These are transparent subsets, not proof of production accuracy or a claim of state-of-the-art performance. Citation validation proves quote provenance, not semantic entailment; ambiguous scope, wrong units, OCR and unsupported questions still need review.

**0.3 is a breaking architecture simplification.** Reindex original files into SQLite; keep old checkpoints/stores for rollback. See [migration and operations](docs/OPERATIONS.md), [architecture](docs/ARCHITECTURE.md), [CLI](docs/CLI.md), and [security](SECURITY.md). The notebook is a small optional interface over this API.

Dependabot version-update PRs remain disabled. CI uses read-only permissions and never writes branches. Repository-level automatic security-update settings require owner access and remain unverified; [operations](docs/OPERATIONS.md) explains the separate setting.

MIT licensed. External benchmark sources retain their own attribution/license.
