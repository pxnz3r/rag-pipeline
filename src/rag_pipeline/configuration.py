"""Explicit provider factories; configuration contains no implicit model choices.

Python factories are operator-trusted installed code, never downloaded from a
model repository or imported from retrieved documents.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path


def load_config(path):
    raw = Path(path).read_bytes()
    if len(raw) > 65536:
        raise ValueError("Pipeline configuration exceeds 64 KiB")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Non-finite configuration number")

    config = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(config, dict) or set(config) - {
        "embedding",
        "reranker",
        "generation",
        "search",
        "answer",
    }:
        raise ValueError("Unknown pipeline configuration section")
    for name, value in config.items():
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section {name} must be an object")
    for name, keys in {
        "search": {
            "k",
            "candidates",
            "min_cosine",
            "context_chars",
            "diversify",
            "distinct_documents",
            "mode",
            "match",
        },
        "answer": {"max_evidence_chars", "context_order", "reasoning_constants"},
    }.items():
        if set(config.get(name, {})) - keys:
            raise ValueError(f"Unknown {name} configuration option")
    return config


def provider(spec, role):
    if not isinstance(spec, dict) or set(spec) - {"kind", "factory", "options"}:
        raise ValueError("Provider requires kind and explicit options")
    options = spec.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("Provider options must be an object")
    kind = spec.get("kind")
    if kind == "python":
        target = spec.get("factory", "")
        if not isinstance(target, str) or ":" not in target:
            raise ValueError("Python provider requires installed module:factory")
        module, name = target.split(":", 1)
        if not module or not name.isidentifier():
            raise ValueError("Invalid provider factory")
        factory = getattr(importlib.import_module(module), name)
    elif kind == "openai":
        from .providers import ChatServer, EmbeddingServer, RerankingServer

        factory = {
            "embedding": EmbeddingServer,
            "reranker": RerankingServer,
            "generation": ChatServer,
        }[role]
    elif kind == "onnx" and role in {"embedding", "reranker"}:
        from .embeddings import ONNXEmbedding, ONNXReranker

        factory = ONNXEmbedding if role == "embedding" else ONNXReranker
    else:
        raise ValueError(f"Unsupported {role} provider kind")
    if kind != "python" and "factory" in spec:
        raise ValueError("factory is only valid for Python providers")
    instance = factory(**options)
    required = {"embedding": "encode", "reranker": "score", "generation": "__call__"}[
        role
    ]
    if not callable(getattr(instance, required, None)):
        raise ValueError(f"Provider lacks {required} interface")
    if role != "generation" and (
        not isinstance(getattr(instance, "signature", None), str)
        or not instance.signature.strip()
    ):
        raise ValueError(
            "Retrieval providers require a stable representation signature"
        )
    return instance
