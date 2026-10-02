from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    pipeline_version: str = "2.0"
    groq_timeout: float = 20.0
    retrieval_top_k: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 20
    final_top_k: int = 5
    rerank_batch_size: int = 16
    context_max_chars: int = 2000
    eval_throttle_sec: float = 0.5
    min_chunk_chars: int = 100
    gc_interval_pdfs: int = 50
    max_checkpoint_bytes: int = 500 * 1024 * 1024
    chunk_max_chars: int = 1800
    chunk_overlap_chars: int = 200
    max_pdf_bytes: int = 100 * 1024 * 1024
    max_pdf_pages: int = 10000
    generation_model: str = "llama-3.3-70b-versatile"
    dense_embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-large"

    def __post_init__(self) -> None:
        for name in (
            "retrieval_top_k",
            "rrf_k",
            "rerank_candidates",
            "final_top_k",
            "rerank_batch_size",
            "context_max_chars",
            "min_chunk_chars",
            "gc_interval_pdfs",
            "max_checkpoint_bytes",
            "chunk_max_chars",
            "max_pdf_bytes",
            "max_pdf_pages",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.chunk_overlap_chars < self.chunk_max_chars:
            raise ValueError(
                "chunk_overlap_chars must be between 0 and chunk_max_chars - 1"
            )
        if self.min_chunk_chars > self.chunk_max_chars:
            raise ValueError("min_chunk_chars must not exceed chunk_max_chars")
        for name in ("groq_timeout", "eval_throttle_sec"):
            value = getattr(self, name)
            if (
                not math.isfinite(value)
                or value < 0
                or (name == "groq_timeout" and value == 0)
            ):
                raise ValueError(f"{name} must be finite and within its valid range")
        for name in (
            "pipeline_version",
            "generation_model",
            "dense_embedding_model",
            "reranker_model",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def processed_dir(self) -> Path:
        return self.base_dir / "processed_data"

    @property
    def db_dir(self) -> Path:
        return self.base_dir / "chroma_db"

    @property
    def lightrag_dir(self) -> Path:
        return self.base_dir / "lightrag_index"

    def ensure_directories(self) -> None:
        for d in (self.data_dir, self.processed_dir, self.db_dir, self.lightrag_dir):
            d.mkdir(parents=True, exist_ok=True, mode=0o700)


def load_settings(base_dir: Path | str | None = None) -> Settings:
    root = (
        Path(base_dir)
        if base_dir is not None
        else Path(os.environ.get("RAG_BASE_DIR", "."))
    )
    settings = Settings(
        base_dir=root,
        pipeline_version=os.environ.get("PIPELINE_VERSION", "2.0"),
        groq_timeout=_env_float("GROQ_TIMEOUT", 20.0),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 50),
        rrf_k=_env_int("RRF_K", 60),
        rerank_candidates=_env_int("RERANK_CANDIDATES", 20),
        final_top_k=_env_int("FINAL_TOP_K", 5),
        rerank_batch_size=_env_int("RERANK_BATCH_SIZE", 16),
        context_max_chars=_env_int("CONTEXT_MAX_CHARS", 2000),
        eval_throttle_sec=_env_float("EVAL_THROTTLE_SEC", 0.5),
        min_chunk_chars=_env_int("MIN_CHUNK_CHARS", 100),
        gc_interval_pdfs=_env_int("GC_INTERVAL_PDFS", 50),
        max_checkpoint_bytes=_env_int("MAX_CHECKPOINT_BYTES", 500 * 1024 * 1024),
        chunk_max_chars=_env_int("CHUNK_MAX_CHARS", 1800),
        chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", 200),
        max_pdf_bytes=_env_int("MAX_PDF_BYTES", 100 * 1024 * 1024),
        max_pdf_pages=_env_int("MAX_PDF_PAGES", 10000),
        generation_model=os.environ.get("GENERATION_MODEL", "llama-3.3-70b-versatile"),
        dense_embedding_model=os.environ.get(
            "DENSE_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"
        ),
        reranker_model=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-large"),
    )
    settings.ensure_directories()
    return settings
