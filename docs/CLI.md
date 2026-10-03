# CLI

Global `--index PATH` defaults to `processed_data/index.sqlite`. `--config FILE` selects explicit providers and search/answer budgets. No model is selected by default. `--embedding module:factory` and `--reranker module:factory` accept installed adapters without a fixed name list. `--dense`/`--rerank` require a supplied provider; chat generation requires a configuration and `--generate`. Place global flags before the command. See [configuration and migration](CONFIGURATION.md).

`prepare-reranker` prepares the chosen token adapter's passage vectors; `evaluate --cache-reranker` reports that cost separately. `--vector-cache-mib 64` bounds the unfiltered dense matrix; `0` streams. Server embedding flags accept an explicit endpoint, model and immutable revision, with `--query-template`/`--passage-template`; plain text is the default. Network adapters send text to your chosen server. Local ONNX adapters do not send inference text to a model server.

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
