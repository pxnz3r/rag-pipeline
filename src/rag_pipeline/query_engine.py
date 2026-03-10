from __future__ import annotations

import asyncio
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .retrieval import reciprocal_rank_fusion, top_k_indices_desc


async def generate_trading_answer_robust(
    query: str,
    collection,
    rag_instance=None,
    *,
    bm25_index=None,
    bm25_id_map: dict[int, str] | None = None,
    retrieval_top_k: int = 50,
    rrf_k: int = 60,
    rerank_candidates: int = 20,
    rerank_batch_size: int = 16,
    final_top_k: int = 5,
    context_max_chars: int = 2000,
    tokenize_for_bm25_fn: Callable[[str], list[str]] | None = None,
    get_eval_components_fn: Callable[[], tuple[Any, Any, Any]] | None = None,
    get_cached_groq_client_fn: Callable[[], Any] | None = None,
    query_param_factory: Callable[[], Any] | None = None,
    logger=None,
) -> tuple[str, list[str]]:
    if not query or not str(query).strip():
        return "Empty Query", []

    global_context = ""
    if rag_instance:
        try:
            query_param = query_param_factory() if query_param_factory else None
            if query_param is not None:
                global_context = await rag_instance.query(query, param=query_param)
            else:
                global_context = await rag_instance.query(query)
        except (RuntimeError, ValueError) as exc:
            if logger:
                logger.error("LightRAG Error: %s", exc)

    local_ctx: list[str] = []
    try:
        dense_results = collection.query(query_texts=[query], n_results=retrieval_top_k)
        dense_ids = dense_results["ids"][0]

        bm25_top_ids: list[str] = []
        if bm25_index is not None and bm25_id_map:
            if tokenize_for_bm25_fn is None:
                raise ValueError(
                    "tokenize_for_bm25_fn is required when BM25 index is supplied."
                )
            tokenized_query = tokenize_for_bm25_fn(query)
            bm25_scores = bm25_index.get_scores(tokenized_query)
            bm25_top_idx = top_k_indices_desc(np.asarray(bm25_scores), retrieval_top_k)
            bm25_top_ids = [
                bm25_id_map[int(i)] for i in bm25_top_idx if int(i) in bm25_id_map
            ]
        elif bm25_index is None and logger:
            logger.warning(
                "BM25 Index not initialized. Proceeding with Dense-only retrieval."
            )

        hybrid_scores = reciprocal_rank_fusion(dense_ids, bm25_top_ids, k=rrf_k)
        top_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[
            :rerank_candidates
        ]

        top_docs = []
        if top_ids:
            top_results = collection.get(ids=top_ids, include=["documents"])
            top_docs = top_results.get("documents") or []
        if top_docs and isinstance(top_docs[0], list):
            top_docs = top_docs[0]
        if logger:
            logger.info(
                "Retrieved %s docs for %s requested IDs", len(top_docs), len(top_ids)
            )

        if top_docs:
            if get_eval_components_fn is None:
                raise ValueError("get_eval_components_fn is required for reranking.")
            _, _, reranker = get_eval_components_fn()
            pairs = [[query, doc] for doc in top_docs]
            scores = reranker.predict(pairs, batch_size=rerank_batch_size)
            top_indices = top_k_indices_desc(np.asarray(scores), final_top_k)
            local_ctx = [top_docs[int(i)] for i in top_indices]

    except (RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
        if logger:
            logger.error("Retrieval Error: %s", exc)

    local_ctx_truncated = [ctx[:context_max_chars] for ctx in local_ctx]
    local_str = "\n---\n".join(local_ctx_truncated)
    prompt = f"""Answer based on contexts.
    GLOBAL: {global_context}
    LOCAL: {local_str}
    Query: {query}"""

    try:
        if get_cached_groq_client_fn is None:
            raise ValueError(
                "get_cached_groq_client_fn is required for answer generation."
            )
        client = get_cached_groq_client_fn()
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return resp.choices[0].message.content, local_ctx
    except Exception as exc:  # noqa: BLE001 - API client exception surface varies by runtime
        return f"Gen Error: {exc}", local_ctx


async def run_evaluation_pipeline(
    questions: Sequence[str],
    ground_truths: Sequence[str],
    collection,
    rag_instance,
    *,
    generate_answer_fn,
    dataset_from_dict_fn: Callable[[dict[str, Any]], Any],
    evaluate_fn,
    metrics: Iterable[Any],
    get_eval_components_fn: Callable[[], tuple[Any, Any, Any]],
    throttle_sec: float = 0.5,
    progress_fn: Callable[[Sequence[str]], Iterable[str]] = lambda x: x,
) -> Any:
    answers, contexts = [], []
    for question in progress_fn(questions):
        ans, ctx = await generate_answer_fn(question, collection, rag_instance)
        answers.append(ans)
        contexts.append(ctx)
        await asyncio.sleep(throttle_sec)

    dataset = dataset_from_dict_fn(
        {
            "question": list(questions),
            "answer": answers,
            "contexts": contexts,
            "ground_truth": list(ground_truths),
        }
    )
    groq_evaluator, hf_embeddings, _ = get_eval_components_fn()
    return evaluate_fn(
        dataset=dataset,
        metrics=list(metrics),
        llm=groq_evaluator,
        embeddings=hf_embeddings,
    )
