# CLI Reference

Executable: `rag-pipeline`

## Commands

### `rag-pipeline test`

Run full pytest suite.

### `rag-pipeline validate-notebook [PATH]`

Validate notebook schema and sanitized Python syntax.  
Default path: `Python3finale.ipynb`.

### `rag-pipeline audit-notebook [PATH]`

Run architecture/pattern checks against the notebook.  
Default path: `Python3finale.ipynb`.

### `rag-pipeline benchmark [--samples N] [--output PATH]`

Run benchmark harness and write JSON metrics output.

### `rag-pipeline smoke [--live]`

- No flag: offline smoke (safe defaults, no paid external calls).
- `--live`: env-gated live prerequisites check (`GROQ_API_KEY`, Ollama health).
