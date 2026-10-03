# Operations and migration

## Install and scope

Python 3.10+ with SQLite FTS5 is required. Install `pip install .`; add `.[models]` for CPU dense/reranking or `.[generation]` for Groq. Default operations use no hosted inference. Model setup downloads pinned public tokenizer/ONNX files; later use loads the local Hugging Face cache. Groq generation explicitly sends the question, retrieved text and metadata to the vendor. Export keys separately; `.env` files are ignored and are not loaded automatically.

Create `data/` explicitly. `rag-pipeline ingest data` synchronizes that **complete directory**, recursively. Empty directories deliberately remove the corpus; absent directories fail. Symlinked files/directories and inaccessible source scans are rejected; no incomplete scan is treated as a complete empty corpus. Use one index per corpus; ingesting another directory into the same index replaces its contents. Scope company/year/jurisdiction/account with adjacent `filename.ext.meta.json` sidecars (up to 32 named short string/integer fields). Account data is retrieved with its row/record labels; metadata sidecars scope a file, not individual CSV fields.

PDFs: 20 MiB input, 1,000 pages, 10 MiB extracted characters. A subprocess has a 45-second wall timeout and POSIX CPU/address-space/output-file caps. Windows only has the parent wall timeout. Resource limits are not a security sandbox: isolate hostile parsing under a separate OS/container boundary before exposing ingestion. Encrypted or textless/scanned PDFs fail and preserve the current index; OCR is not provided. Layout extraction may misorder complex tables, so use labeled CSV/JSON where numeric fidelity matters.

All updates run inside a single durable SQLite transaction. Readers on separate connections retain an old committed snapshot. Parsing/encoding prepares only changed documents in a private on-disk stage. Atomic publication takes the live writer lock after a version check; conflicting peer commits cause an explicit retry error. The busy timeout is 30 seconds and final publications serialize. Pending embedding batches are bounded across files in the staging transaction. Each `Index` connection belongs to its opening thread; serving async requests would require an explicit worker/connection pool. WAL can grow during long readers. New POSIX database/WAL files are owner-only, including under an existing parent directory. Existing file permissions are not changed; keep index/model/source directories private.

Existing schema-3 readers open without a schema-state write, so a long writer does not prevent new readers from opening. Optional token-cache preparation stages inference under a read snapshot, then publishes in one validated writer transaction; reserve enough disk/WAL space for the token arrays and an atomic replacement. The cache is reconstructible and cascades with chunk removal. Warm dense matrix memory defaults to 64 MiB per `Index`, with conservative eligibility accounting; model/native/SQLite allocations are separate. Set `--vector-cache-mib 0` to stream or increase it only after measuring process memory. Filtered queries always stream scoped vectors.

Self-hosted embedding requests send input to the configured endpoint. Verify its tokenizer, pooling and artifact digest before reusing a stored index. Endpoint credentials use `RAG_EMBEDDING_API_KEY`. Network failure or malformed/nonfinite model output aborts the whole ingest rather than retaining mixed vectors. See [modern model and navigation configuration](NAVIGATION.md).

## Upgrade from 0.3

Take a live SQLite backup with `Connection.backup()` before opening the database with 0.4. First open upgrades schema 2→3 atomically: add extractive context, rebuild FTS with identifier-preserving tokenization, retain sources/vectors/generation. An interrupted migration rolls back; missing tables/triggers fail closed. First open can take time on a large index and needs the writer lock. Queries do not require the original source files for this migration.

The next ingest refreshes extraction signatures and rebuilds searchable chunks. Use `ingest --contextual` only when heading-context evaluation supports it. Changing between MiniLM/E5 requires re-embedding the complete corpus; compare using separate `--index` files. Future ingestion/querying must use the matching encoder; lexical-only queries remain offline. For rollback to 0.3, restore the pre-upgrade backup in a separate environment at commit `0545897d3b0871046018e7eb86d7240f58d04ed6`; 0.3 cannot read schema 3. Never downgrade the schema number manually.

## Migration from 0.2

0.3 deliberately replaces the old Python API and developer CLI commands. Keep a copy of original PDFs and 0.2 `processed_data/`, `chroma_db/`, `lightrag_index/`. The new default file is `processed_data/index.sqlite`; no JSON, graph or Chroma state is read or deleted. Reingest originals using `rag-pipeline ingest data`. Add scope metadata and check cited values before relying on results. Install `.[models]` and supply explicit model configuration and run `--dense ingest` to build vectors; future ingests of that index require the same configured embedder. Lexical-only queries still work without model loading.

Changing the embedder/configuration triggers a transactional full rebuild. To remove vectors entirely, create a separate lexical index with `--index another.sqlite`; do not overwrite the original. For rollback, use the 0.2 commit `e217ae819ec40c834205a7344f6c6c91af8193f7` in a separate virtual environment and the retained old stores. Old embeddings/enrichment are not silently trusted or migrated. Never purge retained stores until rollback is no longer needed.

## Backup and health

Use SQLite's `Connection.backup()` for a live consistent backup. Copying only `index.sqlite` while WAL writers run is unsafe. Stop all readers/writers before filesystem copying or restoring an index. `rag-pipeline status` shows counts, generation and the embedding signature. Run `PRAGMA integrity_check` and FTS5's integrity-check command on an owner-controlled connection when diagnosing corruption; never replace a corrupt database with an empty index automatically.

## Automation

Dependabot pip/Actions version PR limits remain zero. CI has only read permissions and cannot write branches. Automatic security updates are a separate GitHub setting: the integration previously returned HTTP 403, so it remains unverified. If those branches recur, the owner can change **Settings → Advanced Security → Dependabot → security updates**. Do not disable CI or advisories to stop version branches. Upgrade dependencies manually, audit resolved extras, run tests and merge only after `test-and-audit` passes.

0.5 removes implicit provider/model choices; see [configuration](CONFIGURATION.md). Staging uses additional temporary disk proportional to changed extracted text/vectors (token preparation proportional to missing tokens), not a full-index copy. Keep sufficient private temporary storage. A reader snapshot during cache encoding can retain WAL pages; long-running workloads need disk monitoring.
