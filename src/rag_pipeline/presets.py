"""Named, measured retrieval stacks: one word instead of a provider configuration.

A preset is an ordinary pipeline configuration (the same schema as ``--config``)
shipped with the package. Naming one is the explicit model choice; nothing is
selected unless a preset or provider is named. The measurements behind each
preset are in docs/TESTING.md and benchmarks/beir-0.7.jsonl.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

_FUSION = {"mode": "hybrid", "fusion": "relative", "dense_weight": 0.6}


def _python(factory):
    return {"kind": "python", "factory": f"rag_pipeline.embeddings:{factory}"}


PRESETS = {
    # BM25 only: no models, no downloads.
    "lexical": {"search": {"mode": "lexical"}},
    # bge-small int8 + BM25: about 3x faster to index than "hybrid", same BEIR
    # and FinQA quality, but lower CUAD clause recall than the old MiniLM route.
    "fast": {"embedding": _python("BGESmall"), "search": dict(_FUSION)},
    # arctic-embed-m int8 + BM25, relative-score fusion weighted 0.6 to dense.
    "hybrid": {"embedding": _python("ArcticM"), "search": dict(_FUSION)},
    # "hybrid", then a gte-reranker cross-encoder over the top 20 chunks.
    # Seconds per query on CPU: worth it when precision matters more than time.
    "rerank": {
        "embedding": _python("ArcticM"),
        "reranker": _python("GTEReranker"),
        "search": {**_FUSION, "rerank_candidates": 20},
    },
}


@dataclass(frozen=True)
class Stack:
    embedder: object
    reranker: object
    search: dict


def config(name: str) -> dict:
    """The preset's configuration, as ``load_config`` would return it."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[name])


def preset(name: str) -> Stack:
    """Instantiate a preset's providers (downloads pinned weights on first use)."""
    from .configuration import provider

    settings = config(name)
    return Stack(
        provider(settings["embedding"], "embedding")
        if "embedding" in settings
        else None,
        provider(settings["reranker"], "reranker") if "reranker" in settings else None,
        settings.get("search", {}),
    )
