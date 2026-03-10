#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_PATTERNS = []

REQUIRED_ANY_PATTERNS = [
    [
        "from rag_pipeline.manifest import (",
        "def load_manifest(store_name: str) -> Dict[str, str]:",
    ],
    [
        "from rag_pipeline.storage import save_json_atomic as save_json_atomic_pkg",
        "def save_json_atomic(data: Any, filepath: Path):",
    ],
    [
        'update_manifest_bulk("chroma", manifest_updates)',
        "from rag_pipeline.chroma_pipeline import populate_chromadb as populate_chromadb_pkg",
    ],
    [
        'batch = collection.get(limit=fetch_size, offset=offset, include=["metadatas"])',
        "from rag_pipeline.chroma_pipeline import purge_stale_data as purge_stale_data_pkg",
    ],
    [
        'remove_manifest_entries("chroma", removed_pdfs)',
        "remove_manifest_entries_fn=lambda store_name, pdf_names: remove_manifest_entries(store_name, pdf_names)",
    ],
    [
        "valid_chunk_ids = {c.id for c in chunks}",
        "from rag_pipeline.enrichment import batch_enrich_chunks as batch_enrich_chunks_pkg",
    ],
]

FORBIDDEN_PATTERNS = [
    'all_data = collection.get(include=["metadatas"])',
    "pdf_chunks = [c for c in chunks_to_ingest if c.pdf_name == pdf]",
]


def code_from_notebook(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def main() -> int:
    notebook_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Python3finale.ipynb")
    )
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    code = code_from_notebook(notebook_path)
    missing = [p for p in REQUIRED_PATTERNS if p not in code]
    missing_any_groups = []
    for options in REQUIRED_ANY_PATTERNS:
        if not any(opt in code for opt in options):
            missing_any_groups.append(options)
    forbidden = [p for p in FORBIDDEN_PATTERNS if p in code]

    if missing or missing_any_groups or forbidden:
        if missing:
            print("Missing required patterns:")
            for p in missing:
                print(f"  - {p}")
        if missing_any_groups:
            print("Missing one-of required pattern groups:")
            for group in missing_any_groups:
                print("  - one of:")
                for p in group:
                    print(f"      * {p}")
        if forbidden:
            print("Forbidden patterns found:")
            for p in forbidden:
                print(f"  - {p}")
        return 1

    print(f"Notebook pattern audit passed: {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
