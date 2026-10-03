# Explicit pipeline configuration — 0.5

The pipeline does not choose an embedding, reranking or generation model. Core search defaults to lexical. Dense/hybrid search requires an explicit embedding provider; missing providers fail instead of silently changing the requested route. Configured roles load only for commands that need them; status and lexical queries do not load an encoder. Generation requires both a configured provider and `ask --generate`.

`rag-pipeline --config pipeline.json --index index.sqlite ingest data` builds an index using its configured encoder. Use the same configuration for search/ask. JSON configuration rejects unknown sections/options, duplicate keys and nonfinite numbers. A CLI search flag overrides its configuration setting. Provider constructor options are validated by that adapter rather than silently discarded.

## Model servers

```json
{
  "embedding": {
    "kind": "openai",
    "options": {
      "endpoint": "http://localhost:8081/v1",
      "model": "YOUR_EMBEDDING_MODEL",
      "revision": "YOUR_IMMUTABLE_ARTIFACT_VERSION",
      "query_template": "{text}",
      "passage_template": "{text}",
      "batch_size": 8
    }
  },
  "reranker": {
    "kind": "openai",
    "options": {
      "endpoint": "http://localhost:8082/v1",
      "model": "YOUR_RERANKING_MODEL",
      "revision": "YOUR_IMMUTABLE_ARTIFACT_VERSION"
    }
  },
  "generation": {
    "kind": "openai",
    "options": {
      "endpoint": "http://localhost:8083/v1",
      "model": "YOUR_GENERATION_MODEL",
      "api_key_env": "MODEL_API_KEY",
      "max_tokens": 3000,
      "json_mode": true
    }
  },
  "search": {
    "mode": "hybrid",
    "k": 5,
    "candidates": 40,
    "min_cosine": 0.3,
    "context_chars": 1800,
    "diversify": true
  },
  "answer": {
    "max_evidence_chars": 12000,
    "context_order": "source"
  }
}
```

Replace placeholders with your actual deployments. These are separate endpoints and model identities, not a required combination or automatic server installer. Omit unused roles. Embeddings use `/embeddings`; reranking uses `/rerank` with indexed `relevance_score` results for every candidate; generation uses `/chat/completions`. Compatible syntax alone does not attest correct weights/tokenization/pooling. Adapter factories support other protocols.

Plain queries/passages are the default. Configure Qwen explicitly with `query_template: "Instruct: Given a question, retrieve relevant passages that answer the question.\nQuery: {text}"`. Configure E5 with `query_template: "query: {text}"` and `passage_template: "passage: {text}"`. Templates accept exactly one plain `{text}` field, no attribute access/conversions. They enter the representation signature. Immutable revisions can be commits, digests or explicit version identifiers; `main`, `master` and `latest` are rejected. This is an operator declaration, not cryptographic server attestation.

Timeout, payload/response, batching and output budgets bound work. Errors exclude server bodies and credentials. Redirects are rejected. Each provider can name its own credential environment variable; credentials are absent from representation signatures. Requests send text to the explicitly configured endpoint. No inference network calls happen in the unconfigured lexical route.

## Generic local ONNX

Use `kind: "onnx"` for embedding/reranking. Explicit options identify the repository `model`, immutable 40-hex `revision`, `weights`, `length`, and optional `batch_size`, `threads`, `providers` and `output_index`. Embedding additionally requires `dimensions` and `pooling` (`mean`, `cls`, `last`, `pooled`), with optional `query_prefix` and `passage_prefix`. Reranking optionally selects `score_index`; multiclass outputs without an explicit score index fail. Unavailable execution providers fail rather than silently falling back to CPU. Output geometry is validated against the configuration. The same graph cannot be interpreted with a different pooling/prefix protocol without changing its signature.

This adapter handles standard token inputs and configured graph outputs. Special token masks, visual inputs, unusual tokenizer protocols and specialized scoring require their own adapter; guessing a model's protocol from its name is unsafe.

## Installed adapters

```json
{
  "embedding": {
    "kind": "python",
    "factory": "my_package.encoders:create_encoder",
    "options": {"model_path": "/models/my-model"}
  },
  "search": {"mode": "hybrid"}
}
```

Factories are explicit operator-trusted installed Python code. Configuration is **not safe to accept from an untrusted client**. Model repositories/retrieved documents cannot select factories. The embedding object supplies `encode(texts)`, optionally `encode_queries(texts)`, and a stable representation `signature`. Rerankers supply `score(question, texts)` and a signature; token caching additionally requires `encode_documents`, `encode_query`, `score_vectors`. Generation returns a callable `(system, evidence_json) -> str` through the same citation validator. The Python API continues to accept these objects directly.

Built-in `MiniLM`, `E5`, `CrossEncoder`, `ColBERT`, `ModernColBERT` remain explicit reproducible reference adapters. Their checkpoint-specific settings do not select or restrict production models. For example, `--embedding rag_pipeline.embeddings:MiniLM --dense ingest data`; querying vectors additionally selects `--mode hybrid`. Any installed factory can occupy that argument. Reranking is analogous: `--reranker rag_pipeline.embeddings:ColBERT prepare-reranker`.

## Upgrade from 0.4

0.5 deliberately removes implicit CLI model choices and named CLI model lists. Replace `--embedding e5` with an explicit factory/configuration, and supply a generator configuration instead of relying on Groq/Llama defaults. The optional Python `groq_generator(model)` now requires a model argument.

Server embedding signatures change to include both templates and the generic protocol. Rebuild a server-encoded index using the same explicitly chosen formatting. Local reference encoder signatures remain compatible, but the next ingest rebuilds chunks once because extraction version 4 fixes metadata/heading shadowing. Ordinary lexical reads of existing indexes continue to work. Changing encoders still requires rebuilding vectors; model agnosticism does not make incompatible vector spaces interchangeable.
