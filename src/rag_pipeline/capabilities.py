"""What this machine and configuration can run, detected, never guessed.

Like CSS feature detection: probe for each capability (an execution provider
that loads, a key that is set, a policy that allows the network) instead of
inferring it from a device or vendor name. The resolver in ``components``
reads only this object, so tests can pass a hand-made one.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

# Hosted services and the environment variable that enables each.
KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "zeroentropy": "ZEROENTROPY_API_KEY",
}
# ONNX Runtime execution providers in preference order; CPU is the floor.
ONNX_PREFERENCE = (
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)
ACCELERATED = set(ONNX_PREFERENCE) - {"CPUExecutionProvider"}
# Non-secret settings that enable components: self-hosted server URLs and the
# model names a hosted chat route should call.
SETTINGS = (
    "RAG_LLAMA_SERVER",
    "RAG_LLAMA_MODEL",
    "RAG_LLAMA_EMBED_SERVER",
    "RAG_LLAMA_EMBED_MODEL",
    "RAG_LLAMA_RERANK_SERVER",
    "RAG_LLAMA_RERANK_MODEL",
    "RAG_OPENAI_MODEL",
    "RAG_GEMINI_MODEL",
)


@dataclass(frozen=True)
class Capabilities:
    cpus: int
    memory_bytes: int
    onnx_providers: tuple[str, ...] = ()
    services: tuple[str, ...] = ()
    settings: tuple[tuple[str, str], ...] = ()
    network: bool = False
    extras: tuple[str, ...] = field(default=())

    def setting(self, name):
        return dict(self.settings).get(name)

    @property
    def accelerator(self):
        """Best accelerated ONNX provider available, or None for CPU only."""
        return next((p for p in self.onnx_providers if p in ACCELERATED), None)

    @property
    def memory_gib(self):
        return self.memory_bytes / 1024**3

    def to_dict(self):
        return {**asdict(self), "accelerator": self.accelerator}


def _memory():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):  # Windows has no sysconf.
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong),
                ("available", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("extended", ctypes.c_ulonglong),
            ]

        status = Status(length=ctypes.sizeof(Status))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.total)


def _installed(module):
    import importlib.util

    return importlib.util.find_spec(module) is not None


def probe(*, network: bool | None = None, environ=os.environ) -> Capabilities:
    """Detect capabilities. ``network`` defaults to ``RAG_ALLOW_NETWORK`` (on).

    Setting ``RAG_ALLOW_NETWORK=0`` keeps every stage local, which is how a
    healthcare or air-gapped deployment opts out of hosted services.
    """
    from .embeddings import _cpus

    if network is None:
        network = environ.get("RAG_ALLOW_NETWORK", "1") != "0"
    providers = ()
    if _installed("onnxruntime"):
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        providers = tuple(p for p in ONNX_PREFERENCE if p in available)
    services = tuple(name for name, key in KEYS.items() if network and environ.get(key))
    extras = tuple(
        name
        for name, module in (
            ("models", "onnxruntime"),
            ("anthropic", "anthropic"),
            ("docling", "docling"),
            ("usearch", "usearch"),
        )
        if _installed(module)
    )
    return Capabilities(
        cpus=_cpus(),
        memory_bytes=_memory(),
        onnx_providers=providers,
        services=services,
        settings=tuple((n, environ[n]) for n in SETTINGS if environ.get(n)),
        network=network,
        extras=extras,
    )
