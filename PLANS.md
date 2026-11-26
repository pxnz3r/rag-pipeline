# Plan: answer accuracy on par with the best RAG products

## What it is for

Question answering over a private document library (filings, contracts, papers, manuals), where every answer carries a checkable citation. Priorities in order: answer correctness, citation provenance, retrieval recall, latency, cost. Nothing about models, vendors, hardware, network or dependencies is fixed ([AGENTS.md](AGENTS.md) "Open choices").

## The biggest problem

The pipeline answers with one shot of retrieval over plain text extracted by pypdf, and it is only measured on retrieval proxies. The products and papers that lead on finance and legal QA win on three things this repo lacks: multi-step agentic retrieval, layout-aware parsing (tables, headings, pages) and LLM-written chunk context. Model swaps, the lever used so far, are the smallest of the four.

Evidence:

- Agentic tool use was the largest single gain in the AgenticRAG paper: FinanceBench answer correctness 92%, against 24.24 for traditional RAG in its Table 3 ([arXiv 2605.05538](https://arxiv.org/abs/2605.05538)).
- PageIndex reports 98.7% on FinanceBench with LLM search over a document tree instead of vectors (self-reported, [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)).
- ColPali found that better ingestion beat a better text embedder on visually rich documents ([arXiv 2407.01449](https://arxiv.org/html/2407.01449v5)).
- Contextual retrieval cut top-20 retrieval failures by 49% with contextual embeddings plus contextual BM25, and by 67% with reranking ([Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)).
- Measured here: the gte reranker added +4.7 NDCG@10 on FiQA and +3.3 points of FinQA recall over the best first stage ([docs/TESTING.md](docs/TESTING.md)).

## Target

| Layer | Now | Target | Why |
| --- | --- | --- | --- |
| Evaluation | Retrieval proxies (BEIR x3, FinQA rows, CUAD spans) | Add answer-level FinanceBench and LegalBench-RAG, judged by exact numeric match plus an LLM judge | Top products compete on answers; retrieval proxies can't rank agentic or parsing changes |
| Parsing | pypdf layout text per page | Docling locally (tables as Markdown, heading tree, page images kept); hosted parser adapter (Reducto or LlamaParse) as an option | Tables and headings decide finance and legal answers; Docling is strong on dense financial tables ([LlamaIndex](https://www.llamaindex.ai/agents/best-document-parsers-for-rag-in-2026), [builderai](https://builderai.tools/blog/pdf-parsing-for-rag-mineru-docling-marker-compared)) |
| Chunking | 900 characters, 100 overlap | Structure-aware chunks: a section or table stays whole up to a token budget; table rows carry their headers | Keeps a clause or table in one retrievable unit |
| Enrichment | Heading context for Markdown only | LLM-written 50 to 100 token chunk context in a derived field, cached by content hash, feeding BM25 and vectors | The 49% failure cut above |
| Lexical | FTS5 BM25, 44 ms on 82k chunks | Keep FTS5 inside the transaction; add a learned sparse channel only if it wins the suite | BM25 is not the quality bottleneck; SPLADE-v3 beat BM25 on 41 of 44 query sets ([arXiv 2403.06789](https://arxiv.org/html/2403.06789v1)) |
| Dense | arctic-embed-m int8 on CPU | Provider tiers: hosted (Voyage 4, Gemini Embedding, Cohere Embed), local GPU (Qwen3-Embedding 0.6B to 8B), CPU fallback (arctic-embed-m) | Qwen3-Embedding and hosted models lead MTEB and RTEB ([Qwen3 report](https://arxiv.org/pdf/2506.05176)) |
| Vector storage | Exact scan of an in-memory float32 matrix | Binary or int8 vectors with float rescoring, plus a usearch HNSW sidecar past about 200k chunks, rebuilt from SQLite and swapped atomically | Binary with rescoring keeps about 95% of quality at 32x less memory ([Hugging Face](https://huggingface.co/blog/embedding-quantization)) |
| Fusion | Relative scores, dense weight 0.6 | Same, with the weight tuned per corpus from an eval set | Measured winner here |
| Rerank | gte-reranker fp32, 3.2 s per query on 4 CPUs | Hosted (Cohere Rerank 4, zerank-2, Voyage rerank) or local GPU Qwen3-Reranker 0.6B/4B; gte on CPU stays the fallback | zerank-2 and Cohere Rerank 4 Pro lead independent reranker tests ([agentset](https://github.com/agentset-ai/reranker-eval), [jev-rerank-bench](https://github.com/anessbelbati/jev-rerank-bench)) |
| Retrieval control | One search, then answer | Agentic loop: an LLM plans, calls search, reads sections, walks the heading tree, filters by metadata, calls the Decimal calculator, and stops on a budget. Citations still go through the quote validator | The largest gain in the literature for finance QA |
| Visual pages | None | ColQwen page-image channel for charts, scans and complex layouts | Text extraction misses non-text evidence ([ViDoRe V3](https://aclanthology.org/2026.acl-long.755.pdf)) |
| Generation | Off by default, Groq adapter | On by default whenever a chat model is configured; providers for Anthropic, OpenAI, Gemini, local llama.cpp | Answers are the product |
| Serving | One-shot CLI, models load per command | Long-running server (HTTP plus MCP) that keeps models and the vector matrix warm; GPU execution providers | Removes per-call model load and cache rebuild |

## Single biggest lever

The agentic retrieval loop on top of layout-aware parsing. The repo already has the pieces: `Navigation` (scoped search, heading tree, bounded reads), the quote and number validator, and the Decimal calculator. What's missing is a strong model driving them and PDFs parsed into a heading tree with tables, so the tree exists for filings and contracts and not only Markdown.

## Guarantees kept

Exact citations with offsets, all-or-nothing ingest with atomic sidecar swaps, and no `eval` of model output. Each is what lets a user trust an answer, so they are product features, not limits.

## Rejected

- **GraphRAG and RAPTOR:** they help global summaries and multi-hop reasoning but often lose to plain RAG on fact lookup, which is most finance and legal QA ([GraphRAG-Bench, ICLR 2026](https://arxiv.org/html/2506.05690v3), [RAG vs GraphRAG](https://arxiv.org/html/2502.11371v2)). Revisit if the eval set gains corpus-wide questions.
- **Replacing SQLite with a vector database:** a sidecar ANN index gives the speed without losing transactional provenance. Revisit past about 10M chunks.
- **LLM listwise reranking as the default:** the evidence is mixed: one small 2026 study (93 queries, single judge) found older cross-encoders beat it, while hosted listwise variants exist. It stays a candidate behind the eval.
- **Late chunking:** +2.7 to 3.6% relative NDCG in its paper ([arXiv 2409.04701](https://arxiv.org/html/2409.04701v3)), smaller than contextual enrichment, and tied to long-context encoders. Try it after phase 3.
- **Long context instead of retrieval:** already exists as `reason --route auto` for small scoped documents; it doesn't scale to a library.

## Order of work

Each phase leaves the app working, lands as its own PR, and closes on its check.

0. **Answer-level eval.** Add FinanceBench (150 questions with evidence pages) and LegalBench-RAG preparation scripts, an answer runner with exact-number match plus an LLM judge, and page-level retrieval recall. Baseline the current `rerank` preset with one strong hosted chat model. Check: `benchmarks/answers-0.8/` holds baseline accuracy with per-question records.
1. **Provider tiers.** Hosted embedding and rerank adapters (Voyage, Cohere, Gemini, OpenAI-compatible), local Qwen3-Embedding and Qwen3-Reranker (ONNX on CPU and CUDA, GGUF through llama.cpp), CUDA and CoreML execution providers, and `quality` and `local` presets. Check: the new presets beat `rerank` on the phase 0 suite, or are recorded as not adopted.
2. **Parsing.** Docling PDF and DOCX parsing into Markdown with tables and a heading tree, page images stored, and structure-aware chunking; bump the extract signature. Check: FinanceBench page recall@10 and CUAD character recall both rise over pypdf.
3. **Contextual enrichment.** LLM chunk context cached by hash in a derived field, used by BM25 and vectors, never cited. Check: retrieval failure at 20 drops on FinanceBench and the BEIR sets; the cost per 1,000 chunks is recorded.
4. **Agentic retrieval.** A tool loop over `Navigation`, search, metadata filters and `calculate`, with a step and token budget, a trace, and verified citations; `ask --agent`. Check: FinanceBench answer accuracy beats the single-shot phase 3 route by a clear margin, with per-question traces.
5. **Server and scale.** An HTTP and MCP server with warm models and matrix; binary or int8 vectors with rescoring; a usearch HNSW sidecar; a 1M-chunk benchmark. Check: p95 hybrid query under 50 ms at 1M chunks on GPU or hosted embeddings, with recall within 1 point of exact search.
6. **Visual channel.** ColQwen page retrieval fused with text. Check: it gains on ViDoRe and on FinanceBench chart questions, or is recorded as not adopted.
7. **Learned sparse and late chunking trials.** SPLADE-v3 or an inference-free sparse encoder as a third channel; late chunking with a long-context encoder. Check: adopted only if the suite improves on balance.

## Cost and privacy trade-offs

Hosted routes send document and query text to the vendor and cost per token; local GPU routes need a GPU; enrichment costs one LLM call per chunk at ingest (cached). Every preset states which of these it uses.
