from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    pipeline_version: str = "1.1"
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
            d.mkdir(parents=True, exist_ok=True)


def load_settings(base_dir: Path | str | None = None) -> Settings:
    root = (
        Path(base_dir)
        if base_dir is not None
        else Path(os.environ.get("RAG_BASE_DIR", "."))
    )
    settings = Settings(
        base_dir=root,
        pipeline_version=os.environ.get("PIPELINE_VERSION", "1.1"),
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
    )
    settings.ensure_directories()
    return settings
