"""Stage components, profile chains and the resolver that picks one per stage.

Every stage has an ordered chain from best to floor. ``resolve`` walks each
chain against detected ``Capabilities`` and takes the first component whose
requirements hold, recording why each better one was skipped. Nothing falls
back silently: the plan is printed by ``rag-pipeline doctor``, stored in the
index, and the encoder (which shapes the index) is fixed per index.

Third-party packages add components through the ``rag_pipeline.components``
entry-point group: each entry point names a callable returning a list of
``Component``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from .capabilities import Capabilities

STAGES = ("encoder", "reranker", "chat")


@dataclass(frozen=True)
class Component:
    name: str
    stage: str
    # Provider configuration (``configuration.provider`` schema), or None for
    # the "none" floor. A callable receives Capabilities, so execution
    # providers can follow the hardware.
    spec: dict | Callable[[Capabilities], dict] | None
    # Returns why this component cannot run here, or None when it can.
    requires: Callable[[Capabilities], str | None] = lambda caps: None
    note: str = ""

    def config(self, caps: Capabilities):
        return self.spec(caps) if callable(self.spec) else self.spec


def _needs_models(caps):
    return None if "models" in caps.extras else "pip install 'rag-pipeline[models]'"


def _needs_service(name):
    def check(caps):
        if not caps.network:
            return "network disabled (RAG_ALLOW_NETWORK=0)"
        return None if name in caps.services else f"no {name} API key"

    return check


def _onnx(factory):
    def spec(caps):
        options = {}
        if caps.accelerator:
            options["providers"] = [caps.accelerator, "CPUExecutionProvider"]
        return {
            "kind": "python",
            "factory": f"rag_pipeline.embeddings:{factory}",
            "options": options,
        }

    return spec


def _all(*checks):
    def check(caps):
        return next((r for c in checks if (r := c(caps))), None)

    return check


def _memory(gib):
    return lambda caps: None if caps.memory_gib >= gib else f"needs {gib} GiB RAM"


def _setting(*names):
    def check(caps):
        missing = [n for n in names if not caps.setting(n)]
        return f"set {', '.join(missing)}" if missing else None

    return check


def _extra(name):
    return lambda caps: (
        None if name in caps.extras else f"pip install 'rag-pipeline[{name}]'"
    )


def _server(kind_options):
    """Spec for an OpenAI-shaped server whose URL or model comes from settings."""

    def spec(caps):
        options = {
            key: caps.setting(value[1:]) if str(value).startswith("$") else value
            for key, value in kind_options.items()
        }
        return {"kind": "openai", "options": options}

    return spec


HOSTED = [
    Component(
        "voyage",
        "encoder",
        {
            "kind": "openai",
            "options": {
                "endpoint": "https://api.voyageai.com/v1",
                "model": "voyage-4-large",
                "revision": "voyage-4-large",
                "api_key_env": "VOYAGE_API_KEY",
                "batch_size": 128,
                "query_options": {"input_type": "query", "output_dimension": 1024},
                "passage_options": {"input_type": "document", "output_dimension": 1024},
            },
        },
        _needs_service("voyage"),
        "Voyage 4 Large, 1024d (hosted; sends text to Voyage)",
    ),
    Component(
        "gemini-embedding",
        "encoder",
        {
            "kind": "python",
            "factory": "rag_pipeline.hosted:GeminiEmbedding",
            "options": {"dimensions": 768},
        },
        _needs_service("gemini"),
        "Gemini Embedding, 768d (hosted; sends text to Google)",
    ),
    Component(
        "zeroentropy",
        "reranker",
        {
            "kind": "openai",
            "options": {
                "endpoint": "https://api.zeroentropy.dev/v1",
                "route": "/models/rerank",
                "model": "zerank-2",
                "revision": "zerank-2",
                "api_key_env": "ZEROENTROPY_API_KEY",
            },
        },
        _needs_service("zeroentropy"),
        "zerank-2 (hosted; sends text to ZeroEntropy)",
    ),
    Component(
        "cohere-rerank",
        "reranker",
        {
            "kind": "openai",
            "options": {
                "endpoint": "https://api.cohere.com",
                "route": "/v2/rerank",
                "model": "rerank-v4.0-pro",
                "revision": "rerank-v4.0-pro",
                "api_key_env": "COHERE_API_KEY",
            },
        },
        _needs_service("cohere"),
        "Cohere Rerank 4 Pro (hosted; sends text to Cohere)",
    ),
    Component(
        "anthropic",
        "chat",
        {"kind": "python", "factory": "rag_pipeline.hosted:AnthropicChat"},
        _all(_needs_service("anthropic"), _extra("anthropic")),
        "Claude Opus 5.5 (hosted; sends evidence to Anthropic)",
    ),
    Component(
        "openai",
        "chat",
        _server(
            {
                "endpoint": "https://api.openai.com/v1",
                "model": "$RAG_OPENAI_MODEL",
                "api_key_env": "OPENAI_API_KEY",
                "token_limit_parameter": "max_completion_tokens",
            }
        ),
        _all(_needs_service("openai"), _setting("RAG_OPENAI_MODEL")),
        "OpenAI chat model named by RAG_OPENAI_MODEL (hosted)",
    ),
    Component(
        "gemini",
        "chat",
        _server(
            {
                "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
                "model": "$RAG_GEMINI_MODEL",
                "api_key_env": "GEMINI_API_KEY",
            }
        ),
        _all(_needs_service("gemini"), _setting("RAG_GEMINI_MODEL")),
        "Gemini chat model named by RAG_GEMINI_MODEL (hosted)",
    ),
]
# llama.cpp's llama-server (CUDA, Metal, Vulkan, HIP, SYCL or CPU) serves one
# model per process; each stage points at its own server.
LLAMA = [
    Component(
        "qwen3-embedding",
        "encoder",
        _server(
            {
                "endpoint": "$RAG_LLAMA_EMBED_SERVER",
                "model": "$RAG_LLAMA_EMBED_MODEL",
                "revision": "$RAG_LLAMA_EMBED_MODEL",
                "instruction": "Given a question, retrieve passages that answer it",
                "batch_size": 32,
            }
        ),
        _setting("RAG_LLAMA_EMBED_SERVER", "RAG_LLAMA_EMBED_MODEL"),
        "Qwen3-Embedding (or any GGUF encoder) on llama-server",
    ),
    Component(
        "qwen3-reranker",
        "reranker",
        _server(
            {
                "endpoint": "$RAG_LLAMA_RERANK_SERVER",
                "model": "$RAG_LLAMA_RERANK_MODEL",
                "revision": "$RAG_LLAMA_RERANK_MODEL",
            }
        ),
        _setting("RAG_LLAMA_RERANK_SERVER", "RAG_LLAMA_RERANK_MODEL"),
        "Qwen3-Reranker (or any GGUF reranker) on llama-server",
    ),
    Component(
        "llama-server",
        "chat",
        _server(
            {
                "endpoint": "$RAG_LLAMA_SERVER",
                "model": "$RAG_LLAMA_MODEL",
                "max_tokens": 4000,
            }
        ),
        _setting("RAG_LLAMA_SERVER", "RAG_LLAMA_MODEL"),
        "Local chat model on llama-server (Qwen3.5 4B or 9B recommended)",
    ),
]

BUILTIN = (
    HOSTED
    + LLAMA
    + [
        Component(
            "arctic-m",
            "encoder",
            _onnx("ArcticM"),
            _all(_needs_models, _memory(4)),
            "arctic-embed-m-v1.5 int8, 768d; best measured local CPU encoder",
        ),
        Component(
            "bge-small",
            "encoder",
            _onnx("BGESmall"),
            _needs_models,
            "bge-small-en-v1.5 int8, 384d; 3x faster indexing, weaker on legal clauses",
        ),
        Component("none", "encoder", None, note="BM25 only"),
        Component(
            "gte-reranker",
            "reranker",
            _onnx("GTEReranker"),
            _all(_needs_models, _memory(4)),
            "gte-reranker-modernbert-base fp32; about 3 s per query on 4 CPU cores",
        ),
        Component("none", "reranker", None, note="fused order"),
        Component("none", "chat", None, note="extractive evidence, no generation"),
    ]
)

# Best first. Hosted and GPU components join these chains as they register;
# a name with no registered component is skipped and reported.
PROFILES = {
    "laptop": {
        "encoder": ["voyage", "gemini-embedding", "arctic-m", "bge-small", "none"],
        "reranker": ["zeroentropy", "cohere-rerank", "none"],
        "chat": ["anthropic", "openai", "gemini", "llama-server", "none"],
    },
    "workstation": {
        "encoder": [
            "voyage",
            "gemini-embedding",
            "qwen3-embedding",
            "arctic-m",
            "bge-small",
            "none",
        ],
        "reranker": [
            "zeroentropy",
            "cohere-rerank",
            "qwen3-reranker",
            "gte-reranker",
            "none",
        ],
        "chat": ["anthropic", "openai", "gemini", "llama-server", "none"],
    },
}
PROFILES["server"] = PROFILES["workstation"]
SEARCH = {"mode": "hybrid", "fusion": "relative", "dense_weight": 0.6}


def registry():
    """All components: built-ins first, then entry-point plugins in name order."""
    from importlib.metadata import entry_points

    found = {}
    for component in BUILTIN + _plugins(entry_points):
        key = (component.stage, component.name)
        if key in found:
            raise ValueError(f"Duplicate {component.stage} component {component.name}")
        found[key] = component
    return found


def _plugins(entry_points):
    components = []
    for point in sorted(
        entry_points(group="rag_pipeline.components"), key=lambda p: p.name
    ):
        loaded = point.load()()
        if not all(isinstance(c, Component) for c in loaded):
            raise ValueError(f"Plugin {point.name} must return Component objects")
        components.extend(loaded)
    return components


def auto_profile(caps: Capabilities):
    """Hardware class from measured capabilities, not from a device name."""
    if caps.accelerator and caps.memory_gib >= 32:
        return "server"
    if caps.accelerator or (caps.cpus >= 8 and caps.memory_gib >= 16):
        return "workstation"
    return "laptop"


@dataclass
class Plan:
    profile: str
    chosen: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    caps: Capabilities | None = None

    def config(self):
        """A pipeline configuration (``load_config`` schema) for this plan."""
        settings = {}
        for stage, key in (("encoder", "embedding"), ("reranker", "reranker")):
            component = self.chosen[stage]
            if component.spec is not None:
                settings[key] = component.config(self.caps)
        chat = self.chosen["chat"]
        if chat.spec is not None:
            settings["generation"] = chat.config(self.caps)
        search = dict(SEARCH) if "embedding" in settings else {"mode": "lexical"}
        if "reranker" in settings:
            search["rerank_candidates"] = 20
        settings["search"] = search
        return settings

    def summary(self):
        return {
            "profile": self.profile,
            "stages": {
                stage: {
                    "use": c.name,
                    "note": c.note,
                    "skipped": dict(self.skipped.get(stage, [])),
                }
                for stage, c in self.chosen.items()
            },
        }

    def signature(self):
        return json.dumps({s: c.name for s, c in self.chosen.items()}, sort_keys=True)


def resolve(profile: str, caps: Capabilities, *, pin: dict | None = None) -> Plan:
    """First runnable component per stage; ``pin`` forces a stage's component.

    A pinned component that cannot run raises instead of falling back: an
    index built with one encoder must be queried with that encoder.
    """
    if profile == "auto":
        profile = auto_profile(caps)
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    components = registry()
    plan = Plan(profile, caps=caps)
    for stage in STAGES:
        chain = [pin[stage]] if pin and stage in pin else PROFILES[profile][stage]
        for name in chain:
            component = components.get((stage, name))
            reason = (
                f"{name} is not installed (no registered component)"
                if component is None
                else component.requires(caps)
            )
            if reason is None:
                plan.chosen[stage] = component
                break
            plan.skipped.setdefault(stage, []).append((name, reason))
        else:
            raise ValueError(
                f"No runnable {stage} component: "
                + "; ".join(f"{n}: {r}" for n, r in plan.skipped[stage])
            )
    return plan
