# RAG Pipeline

[![CI](https://img.shields.io/github/actions/workflow/status/pxnz3r/rag-pipeline/ci.yml?branch=main&label=CI)](https://github.com/pxnz3r/rag-pipeline/actions/workflows/ci.yml)

Local, evidence-first retrieval for financial reports, account records, legal contracts, biomedical abstracts, software documentation and books. One SQLite transaction synchronizes original text, metadata, full-text search and optional vectors. The core has only one third-party dependency (pypdf); NumPy/inference stay in the models extra. Default answers return exact evidence; optional generation must provide verifiable quotes. Numerical reasoning uses explicit source-bound decimal operations.

```bash
python -m pip install --upgrade pip
pip install -e .
mkdir -p data
# Add PDFs, TXT/Markdown, CSV, JSON/JSONL, Python or SQL files to data/.
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
# Optional English retrieval-trained encoder; use a separate index for comparison.
rag-pipeline --index processed_data/e5.sqlite --dense --embedding e5 ingest data --contextual
rag-pipeline --index processed_data/e5.sqlite --dense --embedding e5 search "How much did sales increase?"
rag-pipeline --dense --rerank search "Which obligations survive termination?" --filter jurisdiction=NY
```

Built-in model assets are pinned ONNX files (E5 uses quantized weights); no PyTorch, pickle caches, graph database, or remotely executable model code. Dense search streams scoped vectors or uses a bounded warm matrix cache. Reranking stays opt-in. First use downloads public weights. Built-in retrieval stays on the machine; optional `--embedding-server` sends text to your configured model server. `ask --generate` explicitly sends evidence to Groq and requires `.[generation]` plus `GROQ_API_KEY`.

For token-level matching, use `--reranker colbert prepare-reranker`, then `--rerank --reranker colbert search ...`. Optional `modern-colbert` uses ModernBERT. Passage tokens are prepared once in the same SQLite store, with source/checkpoint validation; this trades disk/indexing work for query latency. Scoped document navigation and source-ordered context support comparisons beyond a static chunk pipeline. See [model and navigation examples](docs/NAVIGATION.md) and the [paper/OSS architecture review](docs/RETRIEVAL-LANDSCAPE.md).

```python
from rag_pipeline import Index, answer

with Index("processed_data/index.sqlite") as index:
    result = answer(
        index, "What was revenue?", filters={"company": "Acme", "year": "2024"}
    )
```

Use `ingest --contextual` to carry extractive Markdown heading hierarchies into both lexical and dense indexing. Results expose `context` separately; cited `text` and offsets always remain original. This adds no hosted-model cost and stays opt-in. Search distinguishes `C++`, `C#` and `snake_case` symbols.

Measured checks include independent FinQA row judgments, CUAD legal character spans, the full BEIR SciFact retrieval test set, authored domain regression canaries, and account/vector scale checks. See [results and reproduction](docs/TESTING.md) and [research](docs/RESEARCH.md). These are transparent subsets, not proof of production accuracy or a claim of state-of-the-art performance. Citation validation proves quote provenance, not semantic entailment; ambiguous scope, wrong units, OCR and unsupported questions still need review.

**0.4 upgrades 0.3 SQLite indexes transactionally on first open**, preserving sources and vectors. Back up before upgrading; 0.3 cannot read the upgraded schema. Enabling heading context or changing the encoder rebuilds the corpus on the next ingest. Users upgrading from 0.2 must reindex original files into SQLite and retain old stores for rollback. See [migration and operations](docs/OPERATIONS.md), [architecture](docs/ARCHITECTURE.md), [CLI](docs/CLI.md), and [security](SECURITY.md). The notebook is a small optional interface over this API.

Dependabot version-update PRs remain disabled. CI uses read-only permissions and never writes branches. Repository-level automatic security-update settings require owner access and remain unverified; [operations](docs/OPERATIONS.md) explains the separate setting.

MIT licensed. External benchmark sources retain their own attribution/license.
