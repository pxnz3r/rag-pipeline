from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable

try:
    import fcntl
except ImportError:
    fcntl = None


@contextlib.contextmanager
def file_lock(filepath: Path):
    if fcntl is None:
        yield
        return
    lock_path = filepath.with_suffix(".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _manifest_path(manifest_dir: Path, store_name: str) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / f"manifest_{store_name}.json"


def _read_manifest(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(path: Path, manifest: Dict[str, str]) -> None:
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp_path.replace(path)


def load_manifest(manifest_dir: Path, store_name: str) -> Dict[str, str]:
    path = _manifest_path(manifest_dir, store_name)
    try:
        with file_lock(path):
            return _read_manifest(path)
    except (OSError, json.JSONDecodeError):
        return {}


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
        manifest = {}
        try:
            manifest = _read_manifest(path)
        except (OSError, json.JSONDecodeError):
            manifest = {}
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
        manifest = {}
        try:
            manifest = _read_manifest(path)
        except (OSError, json.JSONDecodeError):
            manifest = {}
        removed = 0
        for name in names:
            if name in manifest:
                manifest.pop(name, None)
                removed += 1
        if removed:
            _write_manifest(path, manifest)
        return removed
