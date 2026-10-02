# Operations and migration

## Install and scope

Python 3.10+ with SQLite FTS5 is required. Install `pip install .`; add `.[models]` for CPU dense/reranking or `.[generation]` for Groq. Default operations use no hosted inference. Model setup downloads pinned public tokenizer/ONNX files; later use loads the local Hugging Face cache. Groq generation explicitly sends the question, retrieved text and metadata to the vendor. Export keys separately; `.env` files are ignored and are not loaded automatically.

Create `data/` explicitly. `rag-pipeline ingest data` synchronizes that **complete directory**, recursively. Empty directories deliberately remove the corpus; absent directories fail. Symlinked files/directories and inaccessible source scans are rejected; no incomplete scan is treated as a complete empty corpus. Use one index per corpus; ingesting another directory into the same index replaces its contents. Scope company/year/jurisdiction/account with adjacent `filename.ext.meta.json` sidecars (up to 32 named short string/integer fields). Account data is retrieved with its row/record labels; metadata sidecars scope a file, not individual CSV fields.

PDFs: 20 MiB input, 1,000 pages, 10 MiB extracted characters. A subprocess has a 45-second wall timeout and POSIX CPU/address-space/output-file caps. Windows only has the parent wall timeout. Resource limits are not a security sandbox: isolate hostile parsing under a separate OS/container boundary before exposing ingestion. Encrypted or textless/scanned PDFs fail and preserve the current index; OCR is not provided. Layout extraction may misorder complex tables, so use labeled CSV/JSON where numeric fidelity matters.

All updates run inside a single durable SQLite transaction. Readers on separate connections retain an old committed snapshot. Ingestion holds the writer lock while parsing/encoding; the busy timeout is 30 seconds, and only one writer runs at once. Each `Index` connection belongs to its opening thread; serving async requests would require an explicit worker/connection pool. WAL can grow during long readers. Keep index/model/source directories private.

## Migration from 0.2

0.3 deliberately replaces the old Python API and developer CLI commands. Keep a copy of original PDFs and 0.2 `processed_data/`, `chroma_db/`, `lightrag_index/`. The new default file is `processed_data/index.sqlite`; no JSON, graph or Chroma state is read or deleted. Reingest originals using `rag-pipeline ingest data`. Add scope metadata and check cited values before relying on results. Install `.[models]` and run `--dense ingest` to build vectors; future ingests of that index require the same configured embedder. Lexical-only queries still work without model loading.

Changing the embedder/configuration triggers a transactional full rebuild. To remove vectors entirely, create a separate lexical index with `--index another.sqlite`; do not overwrite the original. For rollback, use the 0.2 commit `e217ae819ec40c834205a7344f6c6c91af8193f7` in a separate virtual environment and the retained old stores. Old embeddings/enrichment are not silently trusted or migrated. Never purge retained stores until rollback is no longer needed.

## Backup and health

Use SQLite's `Connection.backup()` for a live consistent backup. Copying only `index.sqlite` while WAL writers run is unsafe. Stop all readers/writers before filesystem copying or restoring an index. `rag-pipeline status` shows counts, generation and the embedding signature. Run `PRAGMA integrity_check` and FTS5's integrity-check command on an owner-controlled connection when diagnosing corruption; never replace a corrupt database with an empty index automatically.

## Automation

Dependabot pip/Actions version PR limits remain zero. CI has only read permissions and cannot write branches. Automatic security updates are a separate GitHub setting: the integration previously returned HTTP 403, so it remains unverified. If those branches recur, the owner can change **Settings → Advanced Security → Dependabot → security updates**. Do not disable CI or advisories to stop version branches. Upgrade dependencies manually, audit resolved extras, run tests and merge only after `test-and-audit` passes.
