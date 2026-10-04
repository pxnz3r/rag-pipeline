# Controlled real-generation numerical QA

This experiment compares retrieved windows and complete original scoped context through the actual pinned FlashRAG sequential class with common adapters. Each primary response contains a source-bound formula and an answer estimate. The string-only estimate has a response-format problem in this fixture and is not a valid arithmetic-quality baseline. The executed program's answer accuracy is reported independently. It is not an official FlashRAG configuration or a FinQA leaderboard submission.

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

Record your actual runtime/hardware in a separate manifest on reruns. Timing here depends on the CPU quota and persistent prompt cache. Resuming a JSONL run requires an identical code/config/data manifest. A changed configuration needs a new output file. The primary process retains its frozen loaded implementation and prompt. Its manifest identifies the exact source version, available at commit `f8c70fd38dfd712e934378be7ffd01a62d7506f0`. Defensive nesting/resource checks were completed after the development process loaded its earlier code. A later fix removes arbitrary ceilings on operator-selected evidence/navigation budgets; the primary still uses 6500 characters. Release verification replays every saved formula against final code and checks prediction/status equality.

## Failed earlier trial

`aborted-pilot.jsonl` preserves ten route records from an earlier 24-case cohort. That trial stopped after duplicate indexed report views exceeded the complete-context budget. Two preceding llama.cpp OOM kills exposed its 8192 MiB default prompt-cache allowance; the final profile bounds that cache to 1024 MiB. The old cohort is excluded from the fresh test at both question and report-page level. Its partial output is not reported as a completed holdout score.

## Interpretation

The judge follows official FinQA five-decimal float rounding and exact numeric gold comparison. Invalid expressions, abstentions and truncations are incorrect. Program equivalence, calibrated unanswerability, full-corpus report discovery and semantic entailment are not scored. A `calculated` result proves operand provenance and arithmetic validity; valid but semantically wrong programs are counted explicitly. Per-route Wilson intervals and case-paired bootstrap differences show uncertainty. Development results cannot establish test performance, and this small CPU fixture cannot establish market leadership or medical/legal correctness.

An exploratory answer-only control is specified while the primary test is in progress, then run sequentially on the same cases and retrieval/context routes. It uses original unannotated source text and a numeric answer schema, with the same model, generation cap and scoring. This assesses the complete structured-program treatment against plain answer generation; its post-hoc status is explicit. No control prompt is selected using individual test answers.

## Completed results and format correction

| Route | Executed primary programs | Numeric-compatible answer-only control |
| --- | ---: | ---: |
| Complete scoped context | 13/24 (54.2%) | 0/24 |
| Retrieved windows | 9/24 (37.5%) | 0/24 |

The control is exploratory, nonthinking and zero-shot, not a strong published configuration. These figures do not establish market leadership or general superiority over other systems. The original string-only control abstained on all 48 calls. The authored `schema-probe.json` instead shows a deeper format mismatch: string-or-null returned brace text on an explicit USD 125 question, while permitting JSON numbers returned 125. The corrected control changes only its answer schema, preserving the original prompt, source windows, budget, model and judge. Primary generation remains frozen. The primary string-only estimates therefore do not establish arithmetic gains. All three trials, manifests and format probes remain visible.

`verification.json` records source versions and checks. `release-replay.json` verifies all 48 primary packing/prediction/status records against release code. `preprocessing-scale.json` measures identical output with bounded quote searches and one-pass annotation assembly; its 8.42× improvement at 50,000 numbers is not a model/generation speedup.

```bash
# Obtain the immutable historical implementation if cloning after squash merge.
git fetch origin refs/pull/26/head:refs/remotes/origin/pr-26
python scripts/benchmark_numeric_evidence.py --old-revision 43a4037 --output /tmp/preprocessing.json
python scripts/replay_qa.py benchmarks/qa-0.6/test.jsonl benchmarks/qa-0.6/test.manifest.json /tmp/replay.json
# Add --answer-only to the QA command for the numeric-compatible control.
```
