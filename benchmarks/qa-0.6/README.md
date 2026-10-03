# Controlled real-generation numerical QA

This experiment compares retrieved windows and complete original scoped context through the actual pinned FlashRAG sequential class with common adapters. Each generated response contains both a source-bound formula and an answer estimate. Comparing these two predictions isolates deterministic execution without another model call. It is not an official FlashRAG configuration or a FinQA leaderboard submission.

The test has 24 numeric cases, with correct report-page identity supplied as input. All 130 earlier query IDs and 112 previously used report-page IDs are excluded. Development uses eight cases. Identical original input views are deduplicated; distinct views are scoped separately. Column labels are repeated per row with the same formatter used for product CSV ingestion. Original numbers are preserved. These inputs are already extracted report text and table matrices, not raw PDF/OCR output.

The frozen profile uses the explicit GGUF listed in `runtime.json`, llama.cpp at the recorded commit, 2 CPU threads, 8192-token server context, 1024 MiB prompt-cache cap, temperature zero, no thinking, a 1000-token generation cap, and 6500 original evidence characters. The local generator is a research fixture, not a production model default. Source files and input digests are recorded in each manifest; per-call output includes original evidence, raw generation, timing, cache tokens, completion tokens and finish reason.

## Reproduce

Install the project's optional CPU models and serve the explicitly chosen checkpoint with the pinned llama.cpp runtime. The server options used here are:

```bash
llama-server -m /path/to/explicit-model.gguf --host 127.0.0.1 --port 8083 \
  -t 2 -tb 2 -c 8192 -b 512 -ub 512 --parallel 1 --cache-ram 1024 \
  --reasoning auto --reasoning-budget 256 --jinja --alias qa-research-model --metrics
```

Obtain the pinned FlashRAG source checkout recorded in `runtime.json`. Then:

```bash
python scripts/prepare_finqa.py /tmp/previous-retrieval.json
python scripts/prepare_qa.py /tmp/qa --exclude /tmp/previous-retrieval.json \
  --test-exclude benchmarks/qa-0.6/prior-cohort-exclusions.json \
  --test-seed 20261004 --exclude-test-reports --table-labels \
  --config benchmarks/qa-0.6/configuration.json
python scripts/benchmark_qa.py --data /tmp/qa --flashrag /path/to/FlashRAG \
  --config benchmarks/qa-0.6/configuration.json --output /tmp/test.jsonl \
  --runtime-manifest benchmarks/qa-0.6/runtime.json --split test \
  --routes retrieval full-context --program-format expression --limit 24
python scripts/summarize_qa.py /tmp/test.jsonl /tmp/test.summary.json
```

Record your actual runtime/hardware in a separate manifest on reruns. Timing here depends on the CPU quota and persistent prompt cache. Resuming a JSONL run requires an identical code/config/data manifest. A changed configuration needs a new output file. The final source algorithm is unchanged during test generation; defensive nesting/resource checks were completed after the development process had loaded its earlier code, which its separate source hashes identify.

## Failed earlier trial

`aborted-pilot.jsonl` preserves ten route records from an earlier 24-case cohort. That trial stopped after duplicate indexed report views exceeded the complete-context budget. Two preceding llama.cpp OOM kills exposed its 8192 MiB default prompt-cache allowance; the final profile bounds that cache to 1024 MiB. The old cohort is excluded from the fresh test at both question and report-page level. Its partial output is not reported as a completed holdout score.

## Interpretation

The judge follows official FinQA five-decimal float rounding and exact numeric gold comparison. Invalid expressions, abstentions and truncations are incorrect. Program equivalence, calibrated unanswerability, full-corpus report discovery and semantic entailment are not scored. A `calculated` result proves operand provenance and arithmetic validity; valid but semantically wrong programs are counted explicitly. Per-route Wilson intervals and case-paired bootstrap differences show uncertainty. Development results cannot establish test performance, and this small CPU fixture cannot establish market leadership or medical/legal correctness.
