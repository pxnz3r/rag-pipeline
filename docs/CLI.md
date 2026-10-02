# CLI

Global `--index PATH` defaults to `processed_data/index.sqlite`. `--dense` loads the pinned MiniLM embedder; `--rerank` loads the optional cross-encoder. Place global flags before the command. All runtime commands work from an installed wheel without a source checkout.

| Command | Behavior |
| --- | --- |
| `ingest DIRECTORY [--size 900 --overlap 100]` | Transactionally synchronize the complete directory; missing input fails |
| `search QUESTION [--filter KEY=VALUE] [-k 5] [--mode hybrid] [--match any]` | JSON sources with document/locator/offset/text/metadata/score |
| `ask QUESTION [search flags] [--generate]` | Default exact evidence; opt-in Groq quote-checked claims |
| `calculate QUESTION --operation OP --operands FILE [search flags]` | Source-bound Decimal sum/difference/ratio/growth_percent |
| `status` | Counts, committed generation and embedding signature |
| `evaluate DATASET [--mode lexical] [-k 5] [--repeats 3] [--split NAME] [--output FILE]` | Judged retrieval, optional character spans, negative cases, index/query timing |

Repeat `--filter` for company/year/jurisdiction/contract/account scope. Keys must be unique; filters are ANDed. `--match all` requires every non-stopword lexical term (useful for exact account lookups); `any` preserves broad natural-language recall. Dense search requires indexed vectors and the matching embedder. Lexical search can read an existing dense index offline. Reranking scores are uncalibrated logits; RRF scores also are not confidence probabilities.

Calculation operand file:

```json
[
  {"source_id":"ledger.csv#row 2:0-74","quote":"Revenue USD millions: 2024=125.0; 2023=100.0.","value":"125.0","unit":"USD millions"},
  {"source_id":"ledger.csv#row 2:0-74","quote":"Revenue USD millions: 2024=125.0; 2023=100.0.","value":"100.0","unit":"USD millions"}
]
```

Use the actual IDs/quotes returned by your search. Growth is `(new−old)/old×100` with a positive base. Mixed units and zero denominators are rejected; no conversion or accounting-semantic inference is performed. Parenthesized amounts are negative. Precision expands for large explicit operands; repeating divisions use the reported precision.

Dataset format: `documents:[{id,text,metadata}]`, `queries:[{id,question,filters,relevant:[document_id],split,spans:[{document,start,end}]}]`. Spans are optional Unicode offsets in the document text. Empty `relevant` marks a negative query. Missing labels/unknown IDs and invalid spans fail; corpus filenames are generated internally rather than accepting dataset paths. Document metrics deduplicate retrieved chunks; character metrics union overlapping returned evidence. These do not measure generated factuality.
