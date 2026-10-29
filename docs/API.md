# Python API

Everything below is importable from `rag_pipeline` and loaded lazily. Validation is strict: invalid arguments raise `ValueError` rather than being coerced.

```python
from rag_pipeline import Index, answer, calculate, Operand, Navigation, reason
```

## `Index`

```python
Index(path, embedder=None, reranker=None, *, vector_cache_bytes=64 * 1024 * 1024)
```

A SQLite-backed corpus. Use it as a context manager, or call `close()`.

| Method | Returns | Notes |
|---|---|---|
| `ingest(root, *, size=900, overlap=100, contextual=False)` | `{"changed", "removed", "documents", "sections", "chunks", "generation", "embedding"}` | Synchronizes the **complete** directory in one transaction. Missing `root` fails; an empty one removes the corpus. `contextual=True` adds Markdown heading context for `.md`/`.txt`. |
| `search(question, *, filters=None, k=5, candidates=40, mode="lexical", min_cosine=0.3, match="any", context_chars=1800, diversify=True, distinct_documents=False)` | `list[Hit]` | `mode` is `"lexical"`, `"dense"` or `"hybrid"` (dense modes need an `embedder`). `filters` are exact metadata matches, ANDed. `match="all"` requires every non-stopword term. |
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
       context_order="ranked", **search_options) -> Answer
```

Without `generate`, returns the exact evidence. With `generate`, calls `generate(system, payload)` and accepts the answer only if every claim quotes the retrieved sources verbatim. `context_order="source"` presents evidence in document order. Extra keyword arguments go to `Index.search`.

`Answer` has `status`, `answer`, `sources` (`list[Hit]`) and `claims`. Status values:

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
Navigation(index, *, filters=None, max_calls=12, max_chars=50000)
```

A budgeted, scoped browsing session for agents. `filters` fix the document scope for every call; `max_calls` and `max_chars` cap tool use. Every call is recorded in `trace`.

| Method | Purpose |
|---|---|
| `documents(*, offset=0, limit=20)` | List documents in scope |
| `outline(document, *, offset=0, limit=20)` | List a document's sections |
| `headings(document, *, parent=None, offset=0, limit=20)` | Browse the heading tree (needs `--contextual` ingest) |
| `read(document, locator, *, start=0, chars=4000)` | Read original text from a section |
| `search(question, **options)` | `Index.search` restricted to the scope |
| `calculate(operation, operands)` | `calculate` over evidence read in this session |
| `answer(question, *, plan, generate=None, max_evidence_chars=12000)` | Answer from the evidence gathered |

See [navigation](NAVIGATION.md).

## `evaluate_rankings`

```python
evaluate_rankings(rankings, relevant_ids, k=5) -> dict
```

Macro recall, precision, MRR and binary NDCG for explicitly judged queries. Missing rankings score zero; empty judgments are rejected. Measures retrieval, not answer factuality. See [testing](TESTING.md).
