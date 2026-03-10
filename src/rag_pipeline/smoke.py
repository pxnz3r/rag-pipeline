from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from .query_engine import generate_trading_answer_robust


@dataclass
class SmokeResult:
    mode: str
    ok: bool
    message: str


class _FakeCollection:
    def query(self, query_texts, n_results):
        return {"ids": [["doc-1", "doc-2"]]}

    def get(self, ids, include):
        return {"documents": ["Synthetic context A", "Synthetic context B"]}


class _FakeReranker:
    def predict(self, pairs, batch_size=16):
        return [0.8, 0.2]


class _FakeClient:
    class Chat:
        class Completions:
            @staticmethod
            def create(messages, model):
                class _Msg:
                    content = "Synthetic answer"

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        completions = Completions()

    chat = Chat()


async def _run_offline_smoke() -> SmokeResult:
    answer, contexts = await generate_trading_answer_robust(
        "What is risk management?",
        _FakeCollection(),
        rag_instance=None,
        bm25_index=None,
        bm25_id_map={},
        retrieval_top_k=5,
        rrf_k=60,
        rerank_candidates=2,
        rerank_batch_size=8,
        final_top_k=1,
        context_max_chars=300,
        tokenize_for_bm25_fn=lambda q: q.split(),
        get_eval_components_fn=lambda: (None, None, _FakeReranker()),
        get_cached_groq_client_fn=lambda: _FakeClient(),
        query_param_factory=None,
        logger=None,
    )
    if not answer or not contexts:
        return SmokeResult(
            mode="offline", ok=False, message="Offline smoke returned empty output."
        )
    return SmokeResult(mode="offline", ok=True, message="Offline smoke passed.")


async def _run_live_smoke() -> SmokeResult:
    missing = []
    if not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if missing:
        return SmokeResult(
            mode="live",
            ok=False,
            message=f"Missing required environment variables for live smoke: {', '.join(missing)}",
        )
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip(
        "/"
    )
    try:
        import requests

        resp = requests.get(f"{ollama_base}/api/tags", timeout=5)
        if resp.status_code != 200:
            return SmokeResult(
                mode="live",
                ok=False,
                message=f"Ollama health check failed: {ollama_base}/api/tags -> {resp.status_code}",
            )
    except Exception as exc:  # noqa: BLE001 - dependency/network/runtime variability in live smoke
        return SmokeResult(
            mode="live",
            ok=False,
            message=f"Live smoke could not reach Ollama at {ollama_base}: {exc}",
        )

    # Live integration is env-gated and intentionally short to avoid accidental cost.
    return SmokeResult(
        mode="live",
        ok=True,
        message="Live smoke prerequisites satisfied (GROQ key + Ollama reachable).",
    )


def run_smoke(*, live: bool = False) -> SmokeResult:
    if live:
        return asyncio.run(_run_live_smoke())
    return asyncio.run(_run_offline_smoke())
