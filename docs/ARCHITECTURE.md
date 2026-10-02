# Architecture (0.3)

```
Local source snapshot → bounded extraction → offsets/chunks → SQLite transaction
                                             ↓                 ↓
                                     optional ONNX vectors   FTS5 + indexed metadata
                                                               ↓
                                      filtered lexical/dense retrieval → RRF
                                                               ↓
                                        optional cross-encoder → original context
                                                               ↓
                                exact evidence / quote-checked generation / Decimal
```

`Index` owns documents, sections, chunks, vectors, metadata lookups and extraction/model signatures. Foreign keys cascade removals, FTS triggers stay in the same transaction, and WAL readers see a committed snapshot while ingestion runs. Invalid parsing, embeddings or writes roll back every changed document. Incremental ingest hashes source/sidecar snapshots; unchanged sources skip parsing and encoding. The index records a generation only when content changes.

`sources` keeps PDF pages, CSV rows and JSON records distinct. CSV headers repeat for every row. JSON number lexemes are preserved as strings in the rendered evidence. Offsets refer to the stored extracted/rendered section, not raw PDF bytes or raw CSV/JSON syntax. Plain text offsets match the original Unicode text. Overlapping chunks prefer line/sentence boundaries, retain exact text and expand locally for clause/table context. Identical basenames in different directories remain separate documents. Queries fetch bounded original text windows through SQLite instead of copying complete sections per candidate. Existing incomplete schemas are never silently rebuilt. NUL-containing extracted text is rejected to preserve window semantics. No synthetic LLM enrichment rewrites evidence.

Metadata values are explicit strings/integers. An indexed key/value table scopes both retrieval channels before ranking; filters are parameterized. Scoping is not authorization: this is a private single-user library, not a tenant-isolated service. Source order/FTS rowids resolve lexical cutoff ties; final fused/reranked ties use chunk IDs. Replacing tied chunks can change cutoff membership.

FTS5 performs lexical retrieval on disk. Dense retrieval normalizes vectors and scans them in 512-row batches, keeping a bounded top-k heap. It is exact O(N×dimension), not ANN or a claim of constant latency at millions of chunks. Optional models use pinned ONNX/tokenizer assets with bounded CPU threads. Dense retrieval is general English, with truncation at 256 tokens; reranking is MS MARCO trained, truncates pairs at 512 tokens, and returns logits, not confidence. Benchmark domain performance before enabling it.

`answer` returns evidence by default and abstains when retrieval returns none. Optional generation receives a separate system instruction and bounded JSON evidence, must return only cited claims, and is rejected on invented source IDs, quotes, or numerical values/currencies/common scales. Numeric formatting may differ while Decimal values must agree; dollar symbols are not assumed to be USD. These checks are conservative and not exhaustive unit/semantic validation. It never displays uncited trailing prose. This proves cited text exists; it cannot prove a paraphrase follows logically, select the correct accounting period, or fully prevent prompt injection. `calculate` requires explicit source IDs, exact quotes, values and matching units; it uses reproducible Decimal operations rather than executing generated code. Operand meaning and scale remain caller responsibilities.

The 0.2 JSON/Chroma/LightRAG manifests, cleanup, graph rebuilds, LangChain/Ragas layers and their tests were removed together. Their split-state synchronization and vulnerable dependency paths are no longer part of the declared runtime. Historical audit evidence is retained; no old user store is automatically deleted. No web fetching, OCR, background branch creation or hidden model/service calls occur during ordinary core ingestion/search.
