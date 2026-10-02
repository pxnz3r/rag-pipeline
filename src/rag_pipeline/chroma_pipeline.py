from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable

from .cleanup import purge_stale_data_streaming, purge_stale_source_ids
from .models import Chunk


def document_fingerprint(chunks: list[Chunk], index_signature: str = "default") -> str:
    """Invalidate indexes when extraction, enrichment, or embedding config changes."""
    digest = hashlib.sha256(json.dumps(index_signature).encode())
    for c in sorted(chunks, key=lambda c: c.id):
        record = (
            c.id,
            c.file_hash,
            c.pipeline_version,
            c.text_with_context or c.text,
            c.page_number,
            c.is_garbled,
        )
        digest.update(b"\n")
        digest.update(json.dumps(record, ensure_ascii=False).encode())
    return "v2:" + digest.hexdigest()


def populate_chromadb(
    *,
    chunks: list[Chunk],
    collection,
    manifest: dict[str, str],
    update_manifest_bulk_fn: Callable[[str, dict[str, str]], None],
    remove_manifest_entries_fn: Callable[[str, list[str]], None],
    logger=None,
    batch_size: int = 50,
    index_signature: str = "default",
) -> object:
    """Synchronize a complete corpus; checkpoint each PDF only after success.

    Requires a single ingestion writer. Upserts are idempotent, so an interrupted
    document is retried. Stale records are deleted only after all replacements
    have succeeded. The backend cannot provide an atomic document transaction;
    readers may see a partial update while ingestion runs.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    groups: dict[str, list[Chunk]] = defaultdict(list)
    seen = set()
    for chunk in chunks:
        if (
            not chunk.id
            or chunk.id in seen
            or not chunk.pdf_name
            or not chunk.file_hash
        ):
            raise ValueError(
                "Chunks must have unique IDs, sources, and nonempty file hashes"
            )
        seen.add(chunk.id)
        groups[chunk.pdf_name].append(chunk)
    for name, group in groups.items():
        if len({(c.file_hash, c.pipeline_version) for c in group}) != 1:
            raise ValueError(f"Inconsistent document versions: {name}")

    for name, group in sorted(groups.items()):
        fingerprint = document_fingerprint(group, index_signature)
        if manifest.get(name) == fingerprint:
            continue
        try:
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                collection.upsert(
                    ids=[c.id for c in batch],
                    documents=[c.text_with_context or c.text for c in batch],
                    metadatas=[
                        {
                            "source": c.pdf_name,
                            "page": c.page_number,
                            "chunk_index": c.chunk_index,
                            "file_hash": c.file_hash,
                            "pipeline_version": c.pipeline_version,
                            "is_garbled": c.is_garbled,
                        }
                        for c in batch
                    ],
                )
            # Use the bounded scanner for this source; do not fetch a whole book.
            purge_stale_source_ids(
                collection, name, {c.id for c in group}, delete_batch_size=batch_size
            )
            update_manifest_bulk_fn("chroma", {name: fingerprint})
        except Exception:
            if logger:
                logger.error(
                    "Index update failed for %s; manifest was not advanced", name
                )
            raise

    # Deletions must run even when no document needs ingestion (including empty corpus).
    for name in sorted(set(manifest) - set(groups)):
        collection.delete(where={"source": name})
        remove_manifest_entries_fn("chroma", [name])
    return collection


def purge_stale_data(collection, *, valid_sources: set[str], logger=None) -> int:
    return purge_stale_data_streaming(
        collection, valid_sources=valid_sources, logger=logger
    )
