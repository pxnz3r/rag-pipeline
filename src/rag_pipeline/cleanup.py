from __future__ import annotations

from typing import Iterable, Iterator


def compute_stale_ids(existing_ids: Iterable[str], new_ids: Iterable[str]) -> list[str]:
    return list(set(existing_ids) - set(new_ids))


def iter_orphan_ids(
    collection, valid_sources: set[str], fetch_size: int = 5000
) -> Iterator[str]:
    offset = 0
    while True:
        batch = collection.get(limit=fetch_size, offset=offset, include=["metadatas"])
        ids = batch.get("ids") or []
        metas = batch.get("metadatas") or []
        if not ids:
            break
        for i, meta in enumerate(metas):
            source = meta.get("source") if isinstance(meta, dict) else None
            if source not in valid_sources and i < len(ids):
                yield ids[i]
        offset += len(ids)
        if len(ids) < fetch_size:
            break


def purge_stale_data_streaming(
    collection,
    valid_sources: set[str],
    fetch_size: int = 5000,
    delete_batch_size: int = 1000,
    logger=None,
) -> int:
    delete_buffer: list[str] = []
    total_deleted = 0

    for orphan_id in iter_orphan_ids(
        collection, valid_sources=valid_sources, fetch_size=fetch_size
    ):
        delete_buffer.append(orphan_id)
        if len(delete_buffer) >= delete_batch_size:
            collection.delete(ids=delete_buffer)
            total_deleted += len(delete_buffer)
            delete_buffer.clear()

    if delete_buffer:
        collection.delete(ids=delete_buffer)
        total_deleted += len(delete_buffer)

    if logger:
        if total_deleted:
            logger.warning("Found and deleted %s orphaned chunks.", total_deleted)
        else:
            logger.info("No orphaned data found.")
    return total_deleted
