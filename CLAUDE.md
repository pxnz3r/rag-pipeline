# CLAUDE.md

@AGENTS.md

## Claude Code notes

- Project subagents live in `.claude/agents/`:
  - `invariant-reviewer`: reviews a diff against the invariants above. Use it before opening a PR that touches `index.py`, `sources.py`, `answers.py`, `reasoning.py` or `expressions.py`.
  - `test-runner`: runs lint and the relevant tests, then reports only failures.
- `main` is protected, so open a PR instead of pushing to it. The `test-and-audit` check must pass before merge.
- Read files in ranges. `index.py` and `reasoning.py` are large, and `benchmarks/*.json` and `docs/dependency-audit-*.json` are generated data that rarely need reading.
