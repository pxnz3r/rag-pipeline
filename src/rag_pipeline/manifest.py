from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable

from filelock import FileLock

from .storage import save_json_atomic


@contextlib.contextmanager
def file_lock(filepath: Path, timeout: float = 30):
    lock_path = filepath.with_name(filepath.name + ".lock")
    with FileLock(str(lock_path), timeout=timeout):
        yield


def _manifest_path(manifest_dir: Path, store_name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", store_name):
        raise ValueError("store_name must contain only letters, digits, '_' or '-'")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / f"manifest_{store_name}.json"


def _read_manifest(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def _write_manifest(path: Path, manifest: Dict[str, str]) -> None:
    save_json_atomic(manifest, path)


def load_manifest(manifest_dir: Path, store_name: str) -> Dict[str, str]:
    path = _manifest_path(manifest_dir, store_name)
    with file_lock(path):
        return _read_manifest(path)


def update_manifest(
    manifest_dir: Path, store_name: str, pdf_name: str, file_hash: str
) -> None:
    update_manifest_bulk(manifest_dir, store_name, {pdf_name: file_hash})


def update_manifest_bulk(
    manifest_dir: Path, store_name: str, updates: Dict[str, str]
) -> None:
    if not updates:
        return
    path = _manifest_path(manifest_dir, store_name)
    with file_lock(path):
        manifest = _read_manifest(path)
        manifest.update(updates)
        _write_manifest(path, manifest)


def remove_manifest_entries(
    manifest_dir: Path, store_name: str, pdf_names: Iterable[str]
) -> int:
    names = list(pdf_names)
    if not names:
        return 0
    path = _manifest_path(manifest_dir, store_name)
    with file_lock(path):
        manifest = _read_manifest(path)
        removed = 0
        for name in names:
            if name in manifest:
                manifest.pop(name, None)
                removed += 1
        if removed:
            _write_manifest(path, manifest)
        return removed
