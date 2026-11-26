# Plan: the best evidence-first answer engine, on any machine

## What it is for

An answer engine over private document libraries for finance, healthcare, law, trading and any other domain, where every answer cites checkable source text. It runs on a laptop with no GPU and no network, on a workstation, and in a data center serving many users. Results scale with the hardware and services available; the engine always runs.

Priorities, in order: answer correctness, citation provenance, recall, latency, cost. Constraints: one maintainer plus AI agents, Python, every capability optional above the floor.

## The biggest problem

The engine is hardwired to a single tier: one process, one SQLite file, CPU ONNX models, one fixed pipeline, one user. There is no model of what a machine can do, no way to pick a better stage when it exists, no domain layer, no server, and no answer-level measurement. The previous plan added strong stages (agentic retrieval, better parsing, hosted models) but never said what happens on a laptop without them, how a data center runs it for many tenants, or how a domain changes the pipeline. Those three are the structural work; the strong stages hang off them.

## Requirements

1. **Capability tiers with declared fallbacks.** Each stage has an ordered chain of implementations from best to floor. A resolver detects hardware, keys, network permission and privacy policy, picks the highest tier that's available and allowed, and records the choice. This is the CSS feature-detection pattern: test for the capability, never guess from the device name.
2. **Fallbacks are visible, never silent.** `rag-pipeline doctor` prints the hardware found and the resolved pipeline. Every answer carries the tiers that produced it. A stage that changes the index format (encoder, parser, enrichment) is fixed per index by its signature and can't fall back at query time. Only query-time stages (reranker, chat model, agent) degrade per request.
3. **Domain packs.** A pack bundles a metadata schema, parsers, query tools, prompts, presets and an eval set, so finance, law, healthcare and trading each get their own pipeline without forks.
4. **Customizable at every layer.** Config layers in order: tier profile, then domain pack, then collection config, then per-query options. Each stage is a registered component with a typed contract, so a user swaps one stage without touching the others.
5. **Same core from laptop to cluster.** Library, CLI, HTTP server and MCP server share one engine. Storage is embedded SQLite on one machine and Postgres for many users, behind one interface that keeps the transaction and provenance guarantees.
6. **Measured per tier and per domain.** Quality and latency are reported for each tier on each domain's eval set, so "results depend on your device" is a table, not a guess.

## Caller's code (the target API)

```python
from rag_pipeline import Library

lib = Library.open("acme", profile="auto", domain="finance")   # resolves tiers once
print(lib.plan)        # parser=docling(cpu) encoder=qwen3-embed-0.6b(coreml) reranker=... llm=qwen3.5-9b(metal)
lib.ingest("filings/")
answer = lib.ask("FY2024 operating margin vs FY2023?", as_of="2025-03-01")
answer.text, answer.citations, answer.trace, answer.tiers
```

```bash
rag-pipeline doctor                                  # hardware, keys, allowed tiers, resolved pipeline
rag-pipeline --profile laptop --domain law ingest contracts/
rag-pipeline --domain trading ask "Guidance changes for NVDA" --as-of 2026-05-01
rag-pipeline serve --http :8080 --mcp                # warm models, many users
```

## Target architecture

| Layer | Now | Recommended (top tier → floor) | Why |
| --- | --- | --- | --- |
| Capability resolver | None; presets name fixed models | Probe RAM, GPU and VRAM, ONNX Runtime execution providers, llama.cpp backends, API keys, network and privacy policy; resolve each stage's chain; `doctor` and `answer.tiers` | Runs everywhere and says what ran |
| Accelerators | ONNX CPU only | ONNX Runtime EPs: CUDA and TensorRT, CoreML, OpenVINO plugin EP, Windows ML (QNN, MIGraphX, TensorRT RTX), DirectML in sustained engineering; llama.cpp for GGUF on CUDA, Metal, Vulkan, HIP, SYCL; CPU floor | Every common device has a fast path ([ORT EPs](https://onnxruntime.ai/docs/execution-providers/), [Windows ML](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers), [llama.cpp](https://github.com/ggml-org/llama.cpp)) |
| Parsing | pypdf text | Hosted (Reducto, LlamaParse) → GPU VLM parser (MinerU) → Docling on CPU → pypdf; XBRL and CSV as structured facts | Tables and headings decide finance and legal answers; ingestion beat a better embedder in ColPali ([LlamaIndex](https://www.llamaindex.ai/agents/best-document-parsers-for-rag-in-2026), [ColPali](https://arxiv.org/html/2407.01449v5)) |
| Chunking | 900 characters | Structure-aware: sections and tables whole up to a token budget, headers repeated on table rows | One clause or table per retrievable unit |
| Enrichment | Markdown headings only | LLM chunk context (hosted → local LLM → headings only), cached by content hash, in a derived field | 49% fewer top-20 failures with contextual embeddings and BM25 ([Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)) |
| Lexical | FTS5 BM25 | FTS5 (SQLite) or Postgres full-text; learned sparse channel only if it wins | BM25 is the floor that never fails; SPLADE-v3 is a measured trial ([arXiv 2403.06789](https://arxiv.org/html/2403.06789v1)) |
| Dense | arctic-embed-m int8 CPU | Hosted (Voyage 4, Gemini Embedding) → GPU Qwen3-Embedding 4B/8B → CPU EmbeddingGemma-300M or arctic-embed-m → none | Top of MTEB and RTEB; EmbeddingGemma leads under 500M parameters and keeps its lead when quantized or truncated ([Qwen3](https://arxiv.org/pdf/2506.05176), [EmbeddingGemma](https://arxiv.org/html/2509.20354v3)) |
| Vectors | float32 matrix, exact scan | Matryoshka truncation plus binary or int8 with rescoring; usearch HNSW sidecar past about 200k chunks; pgvector on the server | About 95% quality at 32x less memory ([Hugging Face](https://huggingface.co/blog/embedding-quantization)) |
| Fusion | Relative scores, weight 0.6 | Same; weight tuned per collection from its eval set; recency term for time-sensitive domains | Measured winner here; time-aware retrieval gained 13 to 27% on financial news ([arXiv 2507.22917](https://arxiv.org/abs/2507.22917v1)) |
| Rerank | gte fp32 CPU, 3.2 s | Hosted (zerank-2, Cohere Rerank 4) → GPU Qwen3-Reranker 4B → CPU Qwen3-Reranker 0.6B or gte → none | Leaders in independent tests ([agentset](https://github.com/agentset-ai/reranker-eval), [jev-rerank-bench](https://github.com/anessbelbati/jev-rerank-bench)) |
| Answering | Optional quote-checked generation | Agentic loop (search, read section, heading tree, metadata filters, calculator, domain tools) driven by: hosted frontier model → local Qwen3.5 9B or 4B → none (extractive evidence) | Agentic tool use was the largest gain on FinanceBench ([arXiv 2605.05538](https://arxiv.org/abs/2605.05538)); Qwen3.5 small models lead local tool calling ([test](https://www.jdhodges.com/blog/local-llms-on-tool-calling-2026-pt1-local-lm/)) |
| Visual pages | None | GPU ColQwen page channel → none | Charts and scans that text misses ([ViDoRe V3](https://aclanthology.org/2026.acl-long.755.pdf)) |
| Storage | One SQLite file | `Store` interface: SQLite embedded (laptop, workstation) and Postgres with pgvector and row-level security (server); sidecars rebuilt and swapped atomically | Keeps transactions and provenance at both ends; pgvector fits to about 10M vectors per collection ([comparison](https://encore.dev/articles/best-vector-databases)) |
| Serving | One-shot CLI | Library, CLI, HTTP server, MCP server over one engine; ingest job queue with workers | Warm models; many users; agents can call it |
| Tenancy and security | Single user | API keys or OIDC, tenant scoping enforced in the store (Postgres RLS), audit log, PHI redaction option; retrieved text treated as untrusted in prompts, read-only tools, step budgets | Healthcare and finance need it; prompt injection is OWASP LLM01 ([OWASP 2025](https://www.hackerone.com/ai/owasp-top-10-llms-2025)) |
| Domains | Generic | Packs: finance (XBRL facts, Decimal calculator, fiscal-period filters), law (citation graph, jurisdiction and date filters, clause types), healthcare (PubMed loader, MedCPT retriever, strict abstention, PHI redaction), trading (point-in-time `as_of`, streaming ingest, recency fusion, market-data tools) | Each domain fails differently |
| Evaluation | Retrieval proxies | Answer-level matrix: FinanceBench, LegalBench-RAG, CUAD, MIRAGE, FinTMMBench or FinTradeBench, plus the current five, each per tier | Proves each tier and domain claim ([MIRAGE](https://arxiv.org/abs/2402.13178), [FinTMMBench](https://arxiv.org/abs/2503.05185v1), [FinTradeBench](https://arxiv.org/pdf/2603.19225)) |

## Single biggest lever

The capability resolver with stage chains. Every other change plugs into it: a new model, parser or LLM becomes one more link in a chain, the laptop and the data center run the same code, and the answer reports which tiers ran. Without it, each new capability is a flag a user must know about, and the low end silently gets nothing.

The single biggest quality lever stays the agentic answer loop over structure-aware parsing. The resolver is what lets it run with a frontier model on a server and a 4B local model on a laptop.

## Guarantees under any tier or store

- Citations are exact source text with offsets; derived data never gets cited.
- Ingest and publication are all-or-nothing; sidecar indexes are rebuilt from the committed store and swapped atomically.
- Model output is never executed; numbers come from Decimal operations.
- An index records the parser, enrichment and encoder that built it and refuses a mismatched query encoder.
- Point-in-time queries never see documents published after `as_of` (no look-ahead in backtests).
- Tenant scoping happens in the store, never per query in application code.
- Hosted tiers run only where the collection's privacy policy allows data to leave the machine.

## Design choices

Two whole shapes were considered for customization:

- **A. Staged engine with a component registry and resolver (chosen).** Fixed stages (parse, chunk, enrich, encode, retrieve, fuse, rerank, answer), each with a typed contract and an ordered chain of registered components. Domain packs register components and config through Python entry points. It hides model and hardware detail behind one contract per stage, and a contributor who adds a component touches one file plus a registration.
- **B. General DAG framework (LangGraph, Haystack style).** Rejected: arbitrary graphs make provenance and signatures hard to guarantee, and the stages here are stable. We accept less freedom in pipeline shape in exchange for guaranteed citations, index signatures and one place to resolve tiers.

## Rejected

- **Rust or C++ rewrite:** the cost is in model compute, already native (ONNX Runtime, llama.cpp, usearch). Revisit only if profiling shows Python overhead above 10% of query time.
- **One vector database for every tier:** a server database is wrong for a laptop. Two stores behind one interface cost more code but keep the floor dependency-free.
- **Vespa or Milvus now:** the right fit past about 100M chunks with complex ranking, but heavy to operate. Revisit when a deployment approaches that.
- **GraphRAG and RAPTOR:** they often lose to plain RAG on fact lookup ([GraphRAG-Bench](https://arxiv.org/html/2506.05690v3)). The law citation graph is a navigation tool, not a GraphRAG index.
- **Silent automatic downgrades:** they hide quality loss. Fallbacks are resolved up front, printed and recorded.
- **LLM listwise reranking as default and late chunking:** mixed or small gains; they stay measured trials.

## Order of work

Each phase leaves the app working, lands as its own PR, and closes on its check.

0. **Answer-level eval matrix.** Preparation scripts for FinanceBench, LegalBench-RAG and a MIRAGE subset; an answer runner with exact numeric match plus an LLM judge; page and span recall; results per tier. Check: `benchmarks/answers-0.8/` has baselines for the `lexical`, `hybrid` and `rerank` presets with one hosted chat model.
1. **Registry, resolver and `doctor`.** Stage contracts, entry-point registry, hardware and key probes, resolved plan stored per index and returned per answer; current presets become tier chains. ONNX execution provider selection (CUDA, CoreML, OpenVINO, DirectML or Windows ML) with CPU as the floor. Check: `doctor` output on this machine matches a recorded fixture; existing benchmarks unchanged; a GPU run where available.
2. **Model tiers.** Hosted embedding, rerank and chat adapters; a llama.cpp server adapter for embeddings, rerank and chat; Qwen3-Embedding, Qwen3-Reranker and EmbeddingGemma; Matryoshka and quantized vectors with rescoring. Check: each tier measured on the phase 0 matrix, and the resolver picks the best measured tier per hardware class.
3. **Parsing tiers and structure-aware chunks.** Docling, MinerU and hosted parser adapters; XBRL and CSV facts; heading trees for PDFs. Check: FinanceBench page recall and CUAD span recall beat pypdf.
4. **Contextual enrichment.** Tiered LLM chunk context, cached by hash. Check: top-20 retrieval failures drop on FinanceBench and BEIR; cost per 1,000 chunks recorded per tier.
5. **Agentic answers.** Tool loop over search, `Navigation`, filters, calculator and domain tools; budgets, traces, injection defenses (untrusted-content markers, read-only tools, quote validation); extractive floor. Check: FinanceBench and LegalBench-RAG answer accuracy beat single-shot at the hosted and local-9B tiers.
6. **Server and scale.** HTTP and MCP server, ingest workers, `Store` interface with Postgres (pgvector, row-level security), auth, audit log, usearch sidecar. Check: two tenants isolated by a store-level test; p95 hybrid query under 100 ms at 1M chunks on the server tier with recall within 1 point of exact search.
7. **Domain packs.** Finance, law, healthcare and trading packs as in the table, each with its eval set and presets. Check: each pack beats the generic pipeline on its own eval; trading `as_of` passes a no-look-ahead test.
8. **Visual channel.** ColQwen page retrieval on GPU tiers. Check: gains on ViDoRe and FinanceBench chart questions, or recorded as not adopted.
9. **Trials.** SPLADE-v3 sparse channel, late chunking, LLM listwise rerank. Check: adopted only where the matrix improves on balance.

## Cost and privacy

Hosted tiers send text to the vendor and cost per token; they run only where the collection allows it. GPU tiers need the hardware. Enrichment costs one LLM call per chunk at ingest, cached. `doctor` and every answer state which of these were used.
