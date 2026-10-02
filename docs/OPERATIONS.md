# Operations Runbook

## Local ingestion (no credentials or model downloads)

```bash
python -m pip install --upgrade pip
python -m pip install -e . -r requirements-dev.txt
mkdir -p ./data
# Place PDFs directly inside ./data.
rag-pipeline ingest --base-dir .
rag-pipeline smoke
```

`--base-dir` contains `data/`, `processed_data/`, `chroma_db/`, and `lightrag_index/`. The PDF directory is the complete corpus snapshot: removing a file removes its chunks on the next successful ingestion. Do not point ingestion at an accidentally empty/wrong directory. Back up state before migration. Encrypted/broken/over-limit PDFs cause a visible failure and preserve the previous master checkpoint.

## Optional notebook adapters

From the repository root:

```bash
python -m pip install -e '.[notebook]'
```

This extra installs large model dependencies. On CPU-only machines, first install a compatible CPU PyTorch wheel using PyTorch's official installation instructions to avoid downloading unnecessary CUDA packages. Install Ollama separately using its official platform installer; the notebook will not download/execute a mutable shell script. Pull `llama3.1` and `nomic-embed-text` before using graph ingestion.

Export `GROQ_API_KEY` before generation/evaluation. PDF extraction and cached BM25 preparation work without it. Set `ENRICH_CONTEXT=1` only when you intend to generate contexts through the API; default cache reuse does not call an API. Graph ingestion and generation remain explicit notebook function calls. Treat sending PDF excerpts to Groq as an external data transfer.

```bash
rag-pipeline smoke --live
```

Live smoke checks the key's presence and Ollama reachability only. It does not validate the key against Groq, download models, or claim full integration success.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAG_BASE_DIR` | `.` | Corpus/state root (CLI `--base-dir` overrides) |
| `PIPELINE_VERSION` | `2.0` | Extraction algorithm version; tuning settings add a signature |
| `CHUNK_MAX_CHARS` | `1800` | Maximum extracted chunk length |
| `CHUNK_OVERLAP_CHARS` | `200` | Adjacent chunk overlap |
| `MIN_CHUNK_CHARS` | `100` | Minimum retained page text length |
| `MAX_PDF_BYTES` | `104857600` | Maximum source file size |
| `MAX_PDF_PAGES` | `10000` | Maximum PDF page count |
| `MAX_CHECKPOINT_BYTES` | `524288000` | Maximum loadable chunk checkpoint size |
| `GROQ_TIMEOUT` | `20` | Groq client network timeout (seconds) |
| `GENERATION_MODEL` | `llama-3.3-70b-versatile` | Answer/enrichment/evaluation model |
| `DENSE_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Chroma embedding model; selects a separate collection |
| `RERANKER_MODEL` | `BAAI/bge-reranker-large` | Cross-encoder reranker |
| `RETRIEVAL_TOP_K` | `50` | Candidates per retrieval channel |
| `RRF_K` | `60` | Fusion rank offset |
| `RERANK_CANDIDATES` | `20` | Candidates to rerank |
| `FINAL_TOP_K` | `5` | Final local contexts |
| `RERANK_BATCH_SIZE` | `16` | Reranker prediction batch size |
| `CONTEXT_MAX_CHARS` | `2000` | Per-context reranking/prompt budget |
| `EVAL_THROTTLE_SEC` | `0.5` | Delay between evaluation questions |
| `GC_INTERVAL_PDFS` | `50` | Garbage collection interval |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Operator-configured Ollama endpoint |

Invalid configuration fails before state directories are created. Changing model IDs can trigger downloads and requires available vendor models. Settings are read at notebook bootstrap; rerun/restart the notebook after changing them so cached clients/models cannot retain old configuration.

## Migration and recovery

1. Back up `processed_data/`, `chroma_db/`, and `lightrag_index/`.
2. Re-run PDF ingestion: extraction signatures change, so old page-sized checkpoints are regenerated into bounded chunks.
3. Optionally re-run enrichment; old-version contexts are not reused.
4. Re-run Chroma ingestion. Legacy manifests reindex once. Collection names now include an embedding-model signature; old collections are retained but are not queried by the new wrapper.
5. Re-run graph ingestion. Legacy append-only graphs are retained; the new active graph lives under `lightrag_index/generations/<generation>/` and is selected by `current.json`.

`RESET_LIGHTRAG_INDEX=1` requests a fresh generation; it no longer recursively deletes the existing graph. Keep enough disk space for a rebuild. After stopping/finalizing old readers, inspect the active pointer before manually removing obsolete generations/collections. Do not delete lock files during an active run.

Corrupt checkpoints/manifests must be restored from a backup or deliberately moved aside before rebuilding. They are not silently treated as an empty corpus. For a failed Chroma ingestion, retry the same complete corpus; successful manifests skip completed PDFs. Wait for sync completion before querying if you require a consistent snapshot.

## Dependency updates and branch automation

Dependabot version-update PRs are disabled (`open-pull-requests-limit: 0` for pip and Actions). CI runs read-only on main pushes and PRs, uses pinned action commits, and audits core dependencies. Update dependencies deliberately, run checks, and refresh the pinned action SHAs after review.

Dependabot security-update PRs are controlled by a separate repository setting. This session's GitHub integration returned HTTP 403 for that setting, so its status could not be verified or changed. If security-update branches still appear, the repository owner can disable **Dependabot security updates** under Settings → Advanced Security (or Code security, depending on GitHub UI). Vulnerability alerts can remain enabled.

The optional notebook stack still has upstream advisories without fixed releases; see [the audit report](AUDIT-2026-10-02.md). Keep Chroma embedded/local, state/cache directories private, and evaluation text-only. Do not expose these dependencies as an untrusted multi-tenant service.
