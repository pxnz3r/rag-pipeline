# CLI

## Tiers, domains and serving

`--profile auto|laptop|workstation|server` resolves the best runnable component for each stage on this machine: hosted services when their API key is set and the network is allowed, llama-server models when `RAG_LLAMA_*` settings point at them, local ONNX models (on CUDA, CoreML, OpenVINO or DirectML when present), down to BM25 and extractive answers. `doctor` prints what was detected and what each stage resolved to, with the reason every better option was skipped. An index records the plan it was built with and keeps that encoder; a query never falls back to a different one. `--profile` and `--preset` are alternatives.

`--domain finance|law|healthcare|trading|general` adds a domain pack: search defaults, answer guidance in the generation prompt, and for healthcare a local-only rule (no hosted stage unless `RAG_ALLOW_NETWORK=1`). `RAG_ALLOW_NETWORK=0` keeps every stage local for any domain.

```bash
rag-pipeline doctor
rag-pipeline --profile auto --domain finance ingest filings/ --parser docling
rag-pipeline --profile auto --domain finance ask "FY2024 operating margin?" --agent
rag-pipeline --profile auto --domain trading search "guidance" --as-of 2024-07-21T09:30
RAG_API_KEYS="acme:<16+ chars>:admin" rag-pipeline --profile auto serve --root /srv/rag
rag-pipeline --profile auto mcp     # stdio MCP server for agents
```

| Command or flag | Behavior |
| --- | --- |
| `doctor` | Capabilities (CPUs, RAM, ONNX execution providers, keys present, settings, network) and the resolved plan |
| `ingest --parser pdfium\|docling` | PDF extractor; part of the index signature, so switching rebuilds |
| `ask --agent [--max-steps 12]` | The chat model searches, reads sections, walks headings, extracts legal citations and calculates before answering; malformed steps are reported back to it; the trace is returned |
| `--as-of DATE` on queries | Only documents whose `published` metadata (ISO date or datetime) is on or before it; undated documents are excluded |
| `serve [--root DIR --host 127.0.0.1 --port 8080]` | HTTP JSON API (`/v1/status`, `/v1/search`, `/v1/ask`, admin `/v1/ingest`); one index per tenant; append-only `audit.jsonl` |
| `mcp` | Model Context Protocol server over stdio with read-only `search`, `list_documents`, `outline`, `read` tools (`[mcp]` extra) |

Settings read from the environment: hosted keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `VOYAGE_API_KEY`, `COHERE_API_KEY`, `ZEROENTROPY_API_KEY`), model names for chat routes without a fixed default (`RAG_OPENAI_MODEL`, `RAG_GEMINI_MODEL`), llama-server endpoints and models (`RAG_LLAMA_SERVER`, `RAG_LLAMA_MODEL`, `RAG_LLAMA_EMBED_SERVER`, `RAG_LLAMA_EMBED_MODEL`, `RAG_LLAMA_RERANK_SERVER`, `RAG_LLAMA_RERANK_MODEL`), `RAG_ALLOW_NETWORK`, and `RAG_API_KEYS` for `serve` (`tenant:key[:admin],...`).

## Commands

Global `--index PATH` defaults to `processed_data/index.sqlite`. `--preset NAME` picks a measured stack in one word: `lexical` (BM25, no models), `fast` (bge-small int8 hybrid), `hybrid` (arctic-embed-m int8 hybrid, the best route measured) or `rerank` (`hybrid` plus a gte-reranker cross-encoder over the top 20 chunks, seconds per query on CPU). Use the same preset for `ingest` and queries. `--config FILE` selects explicit providers and search/answer budgets; its sections replace the preset's sections of the same name. No model is selected unless a preset or provider is named.

```bash
rag-pipeline --preset hybrid ingest data
rag-pipeline --preset hybrid search "Which obligations survive termination?"
rag-pipeline --preset rerank ask "What was 2024 revenue?" --filter company=Acme
```
 `--embedding module:factory` and `--reranker module:factory` accept installed adapters without a fixed name list. `--dense`/`--rerank` require a supplied provider; chat generation requires a configuration and `--generate`. Place global flags before the command. See [configuration and migration](CONFIGURATION.md).

`prepare-reranker` prepares the chosen token adapter's passage vectors; `evaluate --cache-reranker` reports that cost separately. `--vector-cache-mib 512` bounds the in-memory dense matrix that unfiltered and filtered queries share; `0` streams. Server embedding flags accept an explicit endpoint, model and immutable revision, with `--query-template`/`--passage-template`; plain text is the default. Network adapters send text to your chosen server. Local ONNX adapters do not send inference text to a model server.

| Command | Behavior |
| --- | --- |
| `ingest DIRECTORY [--size 900 --overlap 100] [--contextual]` | Transactionally synchronize the complete directory; missing input fails |
| `search QUESTION [--filter KEY=VALUE] [-k 5] [--mode lexical] [--match any]` | JSON sources with document/locator/offset/text/metadata/score/context |
| `ask QUESTION [search flags] [--generate]` | Default exact evidence; opt-in configured generation with quote-checked claims |
| `calculate QUESTION --operation OP --operands FILE [search flags]` | Source-bound Decimal sum/difference/ratio/growth_percent |
| `status` | Counts, committed generation and embedding signature |
| `prepare-reranker` | Transactionally encode missing passage tokens for the selected token adapter |
| `evaluate DATASET [--mode lexical] [-k 5] [--repeats 3] [--split NAME] [--output FILE] [--contextual]` | Judged retrieval, optional character spans, negative cases, index/query timing |

Repeat `--filter` for company/year/jurisdiction/contract/account scope. Keys must be unique; filters are ANDed. `--match all` requires every non-stopword lexical term (useful for exact account lookups); `any` preserves broad natural-language recall. Heading context is an opt-in ingestion configuration, not a per-query summary. `evaluate --contextual` uses the same ingestion path. E5 uses separate query/passage prefixes and a 512-token limit; MiniLM keeps its existing 256-token behavior. Index signatures prevent querying E5 vectors with MiniLM or conversely.

Dense search requires indexed vectors and the matching embedder. Lexical search/status can read an existing dense index without loading an encoder. `ask --context-order source` presents selected evidence in original section order under the same budget. Cross-encoder logits, token MaxSim scores and RRF scores are not confidence probabilities.

Calculation operand file:

```json
[
  {"source_id":"ledger.csv#row 2:0-74","quote":"Revenue USD millions: 2024=125.0; 2023=100.0.","value":"125.0","unit":"USD millions"},
  {"source_id":"ledger.csv#row 2:0-74","quote":"Revenue USD millions: 2024=125.0; 2023=100.0.","value":"100.0","unit":"USD millions"}
]
```

Use the actual IDs/quotes returned by your search. Growth is `(new−old)/old×100` with a positive base. Mixed units and zero denominators are rejected; no conversion or accounting-semantic inference is performed. Parenthesized amounts are negative. Scientific notation is supported with exponent magnitude limited to 1,000 and at most 100 characters per operand. Precision expands for the full exponent range of explicit operands; repeating divisions use the reported precision.

Dataset format: `documents:[{id,text,metadata}]`, `queries:[{id,question,filters,relevant:[document_id],split,spans:[{document,start,end}]}]`. Spans are optional Unicode offsets in the document text. Empty `relevant` marks a negative query. Missing labels/unknown IDs and invalid spans fail; corpus filenames are generated internally rather than accepting dataset paths. Document metrics deduplicate retrieved chunks; character metrics union overlapping returned evidence. These do not measure generated factuality.

`search`, `ask`, `calculate` and `evaluate` accept `--distinct-documents` to return at most one window per source document. Use `-k 10` with this flag for the separately reported SciFact document-ranking protocol; the default retains multiple evidence windows for clause and financial lookup.
