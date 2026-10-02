from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Type

from .models import Chunk


def save_json_atomic(data, filepath: Path, logger=None) -> None:
    """Replace a checkpoint only after serialization and fsync succeed.

    Unique same-directory temporary files avoid writer collisions and symlink
    clobbering. Errors propagate: callers must never acknowledge a failed commit.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=filepath.parent,
            prefix=f".{filepath.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(filepath)
        # Persist the directory entry as well as the file contents on POSIX.
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(filepath.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except (OSError, TypeError, ValueError) as exc:
        if logger:
            logger.error("Failed to save %s: %s", filepath, exc)
        raise
    finally:
        if temp_path is not None and temp_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink()


def get_file_hash(filepath: Path, block_size: int = 65536, logger=None) -> str:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    file_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    except OSError as exc:
        if logger:
            logger.error("Error hashing %s: %s", filepath, exc)
        raise


def load_chunks_map(
    processed_dir: Path,
    filename: str,
    max_checkpoint_bytes: int,
    chunk_type: Type[Chunk] = Chunk,
    logger=None,
) -> Dict[str, Chunk]:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("checkpoint filename must be a plain filename")
    if max_checkpoint_bytes <= 0:
        raise ValueError("max_checkpoint_bytes must be positive")
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
        raise ValueError(f"Checkpoint exceeds size limit: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Checkpoint must be a list of chunk records")
        chunks: Dict[str, Chunk] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Invalid checkpoint chunk record")
            if "pipeline_version" not in item:
                item["pipeline_version"] = "0.0"
            if "is_garbled" not in item:
                item["is_garbled"] = False
            try:
                if "id" in item and "text" in item and "file_hash" in item:
                    chunk = chunk_type(**item)
                    if (
                        not isinstance(chunk.id, str)
                        or not chunk.id
                        or chunk.id in chunks
                    ):
                        raise ValueError("Invalid or duplicate chunk ID")
                    if not all(
                        isinstance(getattr(chunk, key), str)
                        for key in ("text", "pdf_name", "file_hash", "pipeline_version")
                    ):
                        raise ValueError("Invalid checkpoint field type")
                    chunks[chunk.id] = chunk
                else:
                    raise ValueError("Missing required chunk fields")
            except (TypeError, ValueError, AttributeError, KeyError):
                raise ValueError("Invalid checkpoint chunk record") from None
        return chunks
    except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
        if logger:
            logger.error("Corrupted checkpoint file %s: %s", filename, exc)
        raise
