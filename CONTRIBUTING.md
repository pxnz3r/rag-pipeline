# Contributing Guide

Thanks for contributing to `rag-pipeline`.

## Development Setup

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Local Checks

Run all checks before opening a PR:

```bash
rag-pipeline test
rag-pipeline validate-notebook Python3finale.ipynb
rag-pipeline audit-notebook Python3finale.ipynb
rag-pipeline smoke
```

## Branch and Commit Expectations

- Keep PRs focused and small.
- Use clear commit messages (conventional commits preferred).
- Add or update tests for behavior changes.
- Update docs for user-facing or operational changes.

## Notebook + Package Rule

- Keep orchestration in `Python3finale.ipynb`.
- Keep business logic in `src/rag_pipeline`.
- Add tests under `tests/` for package logic and wrapper behavior.

## Pull Request Checklist

- [ ] Tests pass locally.
- [ ] Notebook validation passes.
- [ ] Pattern audit passes.
- [ ] Smoke check passes.
- [ ] Documentation updated (if needed).
