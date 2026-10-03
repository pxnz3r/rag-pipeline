# Document navigation and modern model comparisons

The research review is [RETRIEVAL-LANDSCAPE.md](RETRIEVAL-LANDSCAPE.md). The implementation keeps retrieval, original evidence reads and generated answer validation separate, so models and policies can be compared without rewriting provenance/storage.

## Token retrieval instead of one pooled vector

```bash
pip install -e '.[models]'
rag-pipeline ingest data
# Explicit offline preparation; no dense encoder is needed for lexical + MaxSim.
rag-pipeline --reranker colbert prepare-reranker
rag-pipeline --rerank --reranker colbert search "Which obligations survive termination?"
```

`modern-colbert` selects pinned GTE-ModernColBERT, with a different projection/tokenizer/query configuration and higher CPU/index cost. Both are Apache-2.0 models. Cache preparation retains one checkpoint at a time and rolls back if encoding fails. It stores float32 token arrays in SQLite, deletes them when chunks disappear, and verifies the original passage hash when reading. Changing source files requires ordinary `ingest`, then rerunning preparation; missing tokens remain correct through slower per-query encoding. A cache is optional and reconstructible, not original evidence. It is much larger than a pooled-vector index; check indexing time and disk before choosing it.

Candidate reranking uses at most 40 fused chunks by default. A token reranker cannot find evidence absent from those candidates. This is not a full PLAID/ANN implementation or proof of a published BEIR score. Cross-encoder and token scores are not calibrated answer confidence.

For a fresh controlled comparison, record offline cost separately:

```bash
python scripts/prepare_finqa.py /tmp/finqa.json
rag-pipeline evaluate /tmp/finqa.json --output /tmp/lexical.json
rag-pipeline --rerank --reranker colbert evaluate /tmp/finqa.json --cache-reranker --output /tmp/colbert.json
rag-pipeline --rerank --reranker modern-colbert evaluate /tmp/finqa.json --cache-reranker --output /tmp/modern-colbert.json
```

## Self-hosted instruction-aware embeddings

An OpenAI-compatible `/v1/embeddings` server can replace the built-in pooled encoder. For Qwen3, configure **last-token pooling**, the appropriate tokenizer and an immutable model artifact. The client adds the model's `Instruct: ...\nQuery: ...` query format; passage text stays unprefixed. Server deployment/hardware are independent from this library. An explicit commit or SHA-256 identifies the artifact, but the client cannot attest which weights the server actually runs.

```bash
rag-pipeline --index processed_data/qwen.sqlite --dense \
  --embedding-server http://127.0.0.1:8082/v1 \
  --server-model qwen3-embedding-0.6b \
  --server-revision YOUR_MODEL_SHA256 ingest data
```

Use the same options for subsequent dense/hybrid queries/ingests. Put a real 64-character SHA-256 or 40-character commit in `--server-revision`. Optional endpoint authentication reads `RAG_EMBEDDING_API_KEY`; credentials are not printed/stored in model signatures. Requests have batch/payload/response/time limits and reject invalid indexes/nonfinite vectors. Endpoint calls send passage/query text to that configured service; use a trusted local endpoint when data must stay on the machine. The built-in ONNX routes do not send retrieval text to a hosted model.

## Original document tools

```python
from rag_pipeline import Index, Navigation

with Index("processed_data/index.sqlite") as index:
    session = Navigation(
        index, filters={"company": "Acme"}, max_calls=12, max_chars=50000
    )
    docs = session.documents(limit=10)
    document = docs["documents"][0]["id"]
    sections = session.outline(document)["sections"]
    result = session.read(document, sections[0]["locator"], start=0, chars=4000)
    print(result)
```

`outline` is a paginated directory of original PDF pages/CSV rows/JSON records/text sections, with partial 200-character previews and total lengths. It is **not** an inferred hierarchical table of contents or generated summary. Reads return exact locators/offsets and `next_start` when more source remains. Out-of-scope reads return `not_found`; scope cannot be replaced through a search argument. Pagination, read length, cumulative serialized output and tool calls are bounded. A changed SQLite index invalidates a session rather than silently mixing source versions.

For Markdown ingested with `--contextual`, `session.headings(document)` lists original root headings with exact section ranges; `headings(document, parent=node_id)` drills into children. This hierarchy is stored transactionally beside its source sections, excludes fenced code, preserves names such as C#, and cascades with source removal. A planner can navigate the hierarchy, then read at the returned locator/start offset. It is an extractive ATX-heading tree, not LLM-generated summaries, PDF layout inference or a complete PageIndex reproduction. Inputs without those headings fall back to original section browsing/search.

`session.answer(question, plan=callback, generate=callback)` runs a bounded planner. The planner receives system instructions and JSON history; it returns JSON `{ "action": "read", "arguments": { ... } }`, or documents/outline/search/answer. `answer` has empty arguments. Only read/search outputs become evidence, and final generation goes through the same original-quote/numeric/unit checks as normal answers. `session.trace` records tool actions and output sizes. Planner exceptions/invalid actions fail explicitly; there is no code execution or unrestricted filesystem tool.

Use an actual model callback for quality experiments. Deterministic planner callbacks test this contract, not agent reasoning accuracy. This action surface enables PageIndex/GRASP-style comparisons but does not reproduce their complete index or trained policy. Filters are application scope, not multi-tenant authorization.

The planner may also call `calculate(operation, operands)` over source IDs/quotes it has already read. Available operations are sum, difference, ratio and growth percent; operands include the exact value spelling and an explicit unit present in the quote. The existing Decimal calculator validates every operand. A calculation session finishes with status `calculated` and structured calculation/provenance JSON, rather than allowing a language model to invent a new numeric quote. The model still chooses the accounting item, period and operation: this is arithmetic tooling, not demonstrated FinQA execution accuracy or proof those choices are correct.

## Preserve source order under the same evidence budget

```bash
rag-pipeline ask "Compare the termination provisions" --context-order source
```

This selects/truncates evidence by relevance, then presents those same windows in original document/section order. Numeric page/row ordering follows stored section order, not lexicographic locator sorting. Model answer quality must be measured separately before recommending it over ranked context.
