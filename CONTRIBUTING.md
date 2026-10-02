# Contributing

```bash
pip install -e . -r requirements-dev.txt
ruff check src scripts tests
ruff format --check src scripts tests
pytest
pip-audit --local
```

Core tests use real SQLite/files/subprocesses and run offline on Python 3.10/3.12. Notebook schema and code are covered in the suite. Install `.[models,generation]` and use `RAG_TEST_MODELS=1 pytest` for pinned CPU model integration; audit that resolved environment too.

Keep extraction, retrieval and answer policy in the package. The notebook only calls installed APIs. Preserve exact source provenance and transaction rollback; add boundary regressions for changed behavior. Record independent relevance judgments before tuning and disclose all benchmark scope, exclusions and failures. Avoid redundant wrapper/mocked adapter tests when a small real-storage case verifies the behavior.

PRs must include validation and migration notes for API changes. CI uses read permissions and the protected branch requires `test-and-audit`; do not add unsolicited branch-writing automation.
