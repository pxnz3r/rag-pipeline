from pathlib import Path

from rag_pipeline.manifest import (
    load_manifest,
    remove_manifest_entries,
    update_manifest,
    update_manifest_bulk,
)


def test_manifest_single_and_bulk_updates(tmp_path: Path):
    manifest_dir = tmp_path / "processed_data"

    update_manifest(manifest_dir, "chroma", "a.pdf", "h1")
    update_manifest_bulk(manifest_dir, "chroma", {"b.pdf": "h2", "c.pdf": "h3"})

    manifest = load_manifest(manifest_dir, "chroma")
    assert manifest == {"a.pdf": "h1", "b.pdf": "h2", "c.pdf": "h3"}


def test_remove_manifest_entries(tmp_path: Path):
    manifest_dir = tmp_path / "processed_data"
    update_manifest_bulk(manifest_dir, "chroma", {"a.pdf": "h1", "b.pdf": "h2"})

    removed = remove_manifest_entries(manifest_dir, "chroma", ["b.pdf", "missing.pdf"])
    manifest = load_manifest(manifest_dir, "chroma")

    assert removed == 1
    assert manifest == {"a.pdf": "h1"}
