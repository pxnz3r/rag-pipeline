# Grounded numerical reasoning

`reason` separates source selection, operand selection and calculation. Generation is explicitly configured; there is no production model or vendor default. The model selects references to original numeric tokens and a bounded arithmetic program. A Decimal interpreter validates and executes that program. It never executes model-generated Python.

```bash
rag-pipeline --config pipeline.json reason "What was the revenue growth?" --filter company=Acme --filter year=2024
rag-pipeline --config pipeline.json reason "What was the revenue growth?" --filter company=Acme --route auto
```

The generation provider can use `structured_outputs: true` when the configured endpoint supports OpenAI-style JSON Schema. This capability is explicit because compatible servers differ. The schema constrains operand IDs, operations, argument counts and prior-step reference ranges; execution additionally rejects forward/circular references, invalid source bindings, zero division and excessive numeric resources. Plain installed generator callbacks remain supported and undergo the same execution validation.

```python
from rag_pipeline import Index, reason
from rag_pipeline.configuration import load_config, provider

config = load_config("pipeline.json")
with Index("index.sqlite", provider(config["embedding"], "embedding")) as index:
    result = reason(
        index,
        "What was the revenue growth?",
        generate=provider(config["generation"], "generation"),
        route="auto",
        filters={"company": "Acme"},
        mode="hybrid",
    )
    print(result.status, result.value, result.route, result.steps)
```

Routes:

- `retrieval` searches scoped evidence and packs it within the character budget. This remains the default.
- `context` requires an explicit metadata scope and reads complete original sections in source order. Over-budget context raises an error instead of silently discarding the end of a document.
- `auto` uses complete scoped context when it fits; otherwise it searches under the same scope. This is an explicit size rule, not a learned router or a guarantee that full context answers better.

Results include the actual route, attempt count, original source revisions and an execution trace with resolved source IDs, quotes, values and intermediate Decimal results. `calculated` means the program passed provenance and arithmetic checks. It does **not** certify that the selected quantity, period, units or operation answers the question. `abstained`, `no_evidence` and `program_failed` distinguish insufficient evidence from failed generation/validation. Retries are explicit and bounded (one by default, at most three).

Numeric references are constructed from complete source tokens, including sign, parentheses, exponent and adjacent percentage notation. A copied `100` cannot bind to the prefix of `1000`, nor can a clipped retrieval window supply a new number. Retrieval/read results retain bounded adjacent source context for this lexical validation. Evidence packing preserves that context when it truncates a window. Percentage operands are normalized to fractions by the interpreter; the default prompt requests fractional ratio/percentage answers. The final execution path must include a source operand; unused earlier source steps cannot justify a constant-only result.

`execute_program(program, sources, constants=...)` also supports explicit source leaves for callers building their own plans. Approved named constants can be supplied by the operator; the built-in names are `zero`, `one` and `percent` (100). Supported operations are identity, add, subtract, multiply, divide, bounded real power, min, max, mean and count. There are no implicit currency conversions or inferred dimensional guarantees.

## Compact formulas

`--program-format expression` (Python: `program_format="expression"`) requests a compact mathematical formula instead of the default step array. Both formats use the same source catalog and Decimal executor. `execute_expression("(N1-N0)/N0", sources)` compiles a restricted expression AST into those verified operations; it does not evaluate generated Python. Attribute access, imports, arbitrary indexing, keyword/star arguments and ungrounded numeric literals fail. Bracketed IDs and value-decorated references are supported; displayed values must match the original operand. A literal binds to a source only when its original span is unambiguous; explicit IDs avoid ambiguity. Summation divided by its literal term count compiles to `mean`.

Unicode minus signs, thin/nonbreaking-space thousands separators, scientific notation and leading decimals retain their original source spelling. Adjacent supported units are accepted without treating digits inside ordinary identifiers as operands. Parsing and extraction have explicit resource limits, including normalized CSV/JSON output; a small raw file cannot expand indefinitely through repeated headers or pretty printing.

## Reproducible quality comparison

`prepare_qa.py` selects eight official FinQA development cases and 24 numeric test cases, excluding every query used in the previous retrieval proxy. A further exclusion file and report-level exclusion support a fresh holdout after development trials. Identical report views are indexed once; differing views receive separate scope hashes. `--table-labels` repeats original column labels per row using the CSV formatter, preserving numeric spellings. This is deterministic rendering of an already extracted matrix, not a PDF table reconstruction claim. Sampling uses a fixed seed and numeric answer type; no answer, gold program or gold evidence selects the indexed context. Reports, original tables and subsequent text form the corpus. Official report identity is supplied as scope.

```bash
python scripts/prepare_finqa.py /tmp/prior-retrieval.json
python scripts/prepare_qa.py /tmp/qa --exclude /tmp/prior-retrieval.json --config comparison.json --table-labels
# Obtain the pinned FlashRAG source checkout listed in the result manifest.
python scripts/benchmark_qa.py --data /tmp/qa --flashrag /tmp/FlashRAG --config comparison.json --output /tmp/dev.jsonl --split dev --program-format expression
python scripts/benchmark_qa.py --data /tmp/qa --flashrag /tmp/FlashRAG --config comparison.json --output /tmp/test.jsonl --split test --program-format expression
```

The comparison executes upstream FlashRAG's unchanged sequential and iterative classes with shared adapters. It compares one-pass retrieved context, complete scoped context and two-pass generation-augmented retrieval. The same generator, evidence budget, program prompt and schema apply to all routes. The model's answer estimate and deterministic execution are scored from the **same generation**, isolating the effect of arithmetic execution. This common structured prompt is not FlashRAG's native published prompt/configuration, and the experiment is not a reproduction of a published leaderboard score.

The numeric judge follows FinQA's five-decimal float rounding and exact gold-value comparison. Syntax failures, abstentions and generation truncation count as incorrect predictions. It does not assess official program equivalence. The small subset cannot establish full-test performance, market leadership or cross-domain clinical/legal correctness. Manifests record code/data digests and configured settings; per-call records retain original evidence, raw output, usage, finish reason and timings. Routes are counterbalanced by case on the held-out run; the persistent model's cache can affect latency, so cached tokens and call counts must accompany comparisons. The gold answer field is scored only after predictions are generated; gold programs and evidence labels never enter the generator payload.
