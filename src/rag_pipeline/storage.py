from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Type

from .models import Chunk


def save_json_atomic(data, filepath: Path, logger=None) -> None:
    temp_path = filepath.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(filepath)
    except OSError as exc:
        if logger:
            logger.error("Failed to save %s: %s", filepath, exc)
        if temp_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink()


def get_file_hash(filepath: Path, block_size: int = 65536, logger=None) -> str:
    file_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    except OSError as exc:
        if logger:
            logger.error("Error hashing %s: %s", filepath, exc)
        return ""


def load_chunks_map(
    processed_dir: Path,
    filename: str,
    max_checkpoint_bytes: int,
    chunk_type: Type[Chunk] = Chunk,
    logger=None,
) -> Dict[str, Chunk]:
    filepath = processed_dir / filename
    if not filepath.exists():
        return {}
    if filepath.stat().st_size > max_checkpoint_bytes:
        if logger:
            logger.warning(
                "Checkpoint %s is too large (%s bytes). Skipping load to avoid OOM.",
                filename,
                filepath.stat().st_size,
            )
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks: Dict[str, Chunk] = {}
        for item in data:
            if "pipeline_version" not in item:
                item["pipeline_version"] = "0.0"
            if "is_garbled" not in item:
                item["is_garbled"] = False
            try:
                if "id" in item and "text" in item and "file_hash" in item:
                    chunks[item["id"]] = chunk_type(**item)
            except (TypeError, ValueError, AttributeError, KeyError):
                continue
        return chunks
    except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
        if logger:
            logger.error("Corrupted checkpoint file %s: %s", filename, exc)
        return {}
