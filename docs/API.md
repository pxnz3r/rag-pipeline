# Python API

Everything below is importable from `rag_pipeline` and loaded lazily. Validation is strict: invalid arguments raise `ValueError` rather than being coerced.

```python
from rag_pipeline import Index, answer, calculate, Operand, Navigation, reason, preset
```

## `preset`

```python
stack = preset("hybrid")  # "lexical", "fast", "hybrid" or "rerank"
with Index("processed_data/index.sqlite", stack.embedder, stack.reranker) as index:
    index.ingest("data")
    hits = index.search("What was 2024 revenue?", **stack.search)
```

Returns a `Stack(embedder, reranker, search)`: the preset's providers, instantiated (pinned weights download on first use), and its search options. `rag_pipeline.presets.config(name)` returns the same preset as a configuration dict without loading models.

## `Index`

```python
Index(path, embedder=None, reranker=None, *, vector_cache_bytes=512 * 1024 * 1024)
```

A SQLite-backed corpus. Use it as a context manager, or call `close()`.

| Method | Returns | Notes |
|---|---|---|
| `ingest(root, *, size=900, overlap=100, contextual=False, plan=None, parser="pdfium", enricher=None)` | `{"changed", "removed", "documents", "sections", "chunks", "generation", "embedding", "plan"}` | Synchronizes the **complete** directory in one transaction. Missing `root` fails; an empty one removes the corpus. `contextual=True` adds Markdown heading context for `.md`/`.txt`. `parser="docling"` extracts PDF tables and headings as Markdown. `enricher` (`enrich.ChatEnricher(chat)`) writes a context sentence per chunk, cached beside the index. `plan` records a resolved `components.Plan.summary()`. Parser and enricher enter the index signature. |
| `search(question, *, filters=None, k=5, candidates=40, mode="lexical", min_cosine=0.3, match="any", context_chars=1800, diversify=True, distinct_documents=False, fusion="rrf", dense_weight=0.5, rerank_candidates=None)` | `list[Hit]` | `mode` is `"lexical"`, `"dense"` or `"hybrid"` (dense modes need an `embedder`). `filters` are exact metadata matches, ANDed. `match="all"` requires every non-stopword term. `fusion="rrf"` sums weighted reciprocal ranks (0.5 keeps the classic 1/(60+rank) sum); `"relative"` sums min-max normalized BM25 and cosine scores, weighted by `dense_weight`. A weight of 0 or 1 turns a channel off. A reranker rescores the top `rerank_candidates` fused chunks (default: all `candidates`); the rest keep their fused order below them. |
| `status()` | `dict` | Document/section/chunk counts, publication generation, embedding signature. |
| `prepare_reranker_cache()` | | Stages ColBERT passage tokens for the configured reranker. |
| `close()` | | Closes the database. |

```python
with Index("processed_data/index.sqlite") as index:
    index.ingest("data")
    hits = index.search("revenue", filters={"company": "Acme", "year": "2024"}, k=3)
```

## `Hit`

One retrieved passage. `to_dict()` returns it as a plain dict.

| Field | Meaning |
|---|---|
| `id` | Chunk identifier |
| `document` | Source document id (its file path) |
| `locator` | Section locator, such as a PDF page or CSV row |
| `start`, `end` | Character offsets into the original section |
| `text` | Original source text. This is the only text that can be cited. |
| `context` | Optional derived heading context; never cited |
| `metadata` | Sidecar metadata |
| `score` | Ranking score for this query |
| `source_revision`, `source_prefix`, `source_suffix` | Provenance used to validate quotes |

## `answer`

```python
answer(index, question, *, generate=None, filters=None, max_evidence_chars=12000,
       context_order="ranked", guidance="", **search_options) -> Answer
```

Without `generate`, returns the exact evidence. With `generate`, calls `generate(system, payload)` and accepts the answer only if every claim quotes the retrieved sources verbatim. `context_order="source"` presents evidence in document order. `guidance` appends domain instructions (`domains.get("finance").guidance`). Extra keyword arguments go to `Index.search`, including `as_of="2024-07-21"` for point-in-time retrieval over ISO `published` metadata.

`Answer` has `status`, `answer`, `sources` (`list[Hit]`), `claims` and `detail` (why generation failed validation). Status values:

| Status | Meaning |
|---|---|
| `evidence` | Evidence returned without generation |
| `cited` | Generated answer whose claims passed quote checks |
| `no_evidence` | Nothing relevant was retrieved |
| `abstained` | The generator declined to answer from the evidence |
| `generation_failed` | Generator output was invalid or failed quote checks |

## `calculate` and `Operand`

```python
calculate(operation, operands: list[Operand], sources: list[Hit]) -> dict
Operand(source_id: str, quote: str, value: str, unit: str)
```

Source-bound `Decimal` arithmetic. `operation` is `"sum"`, `"difference"`, `"ratio"` or `"growth_percent"`. Each operand must quote text from the source it names, and units must be compatible (aliases such as `ms`/`milliseconds` normalize). Division by zero and growth from a negative base are rejected.

## Numerical reasoning

```python
reason(index, question, *, generate, max_evidence_chars=12000, max_attempts=1,
       route="retrieval", program_format="steps", constants=None, **search_options) -> ProgramAnswer
reason_from_sources(question, sources, *, generate, max_evidence_chars=12000,
                    max_attempts=1, program_format="steps", constants=None) -> ProgramAnswer
execute_program(program, sources, *, constants=None)
execute_expression(expression, sources, *, constants=None)
```

The generator writes a program (`"steps"`) or a compact formula (`"expression"`). The program is executed by the library, never by `eval`. `route` is `"retrieval"`, `"context"` (read the complete scoped corpus) or `"auto"` (context when it fits the budget, else retrieval). `max_attempts` is 1 to 3.

`ProgramAnswer` has `status`, `value`, `sources`, `steps`, `attempts` and `route`. Status is `calculated`, `no_evidence`, `abstained` or `program_failed`. See [reasoning](REASONING.md).

## `Navigation`

```python
Navigation(index, *, filters=None, max_calls=12, max_chars=50000, search=None)
```

A budgeted, scoped browsing session for agents. `filters` fix the document scope for every call; `max_calls` and `max_chars` cap tool use; `search` holds default search options (a preset's hybrid route). Every call is recorded in `trace`. In `answer`, a malformed model reply, unknown action or bad arguments go back to the model as an error instead of failing (with an allowance equal to the call budget), and running out of steps answers from the evidence gathered.

| Method | Purpose |
|---|---|
| `documents(*, offset=0, limit=20)` | List documents in scope |
| `outline(document, *, offset=0, limit=20)` | List a document's sections |
| `headings(document, *, parent=None, offset=0, limit=20)` | Browse the heading tree (needs `--contextual` ingest) |
| `read(document, locator, *, start=0, chars=4000)` | Read original text from a section |
| `search(question, **options)` | `Index.search` restricted to the scope |
| `citations(document, locator)` | Legal citations (U.S.C., C.F.R., reporters, sections) in a section, with exact offsets |
| `calculate(operation, operands)` | `calculate` over evidence read in this session |
| `answer(question, *, plan, generate=None, max_evidence_chars=12000, guidance="")` | Model-directed browsing, then an answer from the evidence gathered |

## Tiers, domains, server

| Name | Purpose |
|---|---|
| `capabilities.probe(network=None)` | Detected CPUs, RAM, ONNX execution providers, hosted keys, `RAG_*` settings, network policy |
| `components.resolve(profile, caps, pin=None)` | A `Plan` with one component per stage; `.config()` is a pipeline configuration, `.summary()` the record stored in the index |
| `domains.get(name)` | A `Domain` with `guidance`, `metadata`, `search` defaults and `local_only` |
| `server.Engine(root, keys, ...)`, `server.serve(engine, host, port)` | The HTTP server behind `rag-pipeline serve` |
| `mcp_server.build(index, search=None)` | The MCP server behind `rag-pipeline mcp` |

See [navigation](NAVIGATION.md).

## `evaluate_rankings`

```python
evaluate_rankings(rankings, relevant_ids, k=5) -> dict
```

Macro recall, precision, MRR and binary NDCG for explicitly judged queries. Missing rankings score zero; empty judgments are rejected. Measures retrieval, not answer factuality. See [testing](TESTING.md).
