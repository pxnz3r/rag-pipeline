from __future__ import annotations

import json
import tempfile
from typing import Iterable, Iterator


def compute_stale_ids(existing_ids: Iterable[str], new_ids: Iterable[str]) -> list[str]:
    return list(set(existing_ids) - set(new_ids))


def iter_orphan_ids(
    collection, valid_sources: set[str], fetch_size: int = 5000
) -> Iterator[str]:
    if fetch_size <= 0:
        raise ValueError("fetch_size must be positive")
    offset = 0
    while True:
        batch = collection.get(limit=fetch_size, offset=offset, include=["metadatas"])
        ids = batch.get("ids") or []
        metas = batch.get("metadatas") or []
        if not ids:
            break
        for i, chunk_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else None
            source = meta.get("source") if isinstance(meta, dict) else None
            if source not in valid_sources:
                yield chunk_id
        offset += len(ids)
        if len(ids) < fetch_size:
            break


def delete_ids_after_scan(
    collection, ids: Iterable[str], delete_batch_size: int = 1000
) -> int:
    if delete_batch_size <= 0:
        raise ValueError("delete_batch_size must be positive")
    delete_buffer: list[str] = []
    total_deleted = 0

    # Finish pagination before any deletion: deleting while advancing offsets
    # shifts live collection rows and silently skips orphans. Spool IDs to disk
    # so memory remains bounded without relying on a backend snapshot API.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as snapshot:
        for orphan_id in ids:
            snapshot.write(json.dumps(orphan_id) + "\n")
        snapshot.seek(0)
        for line in snapshot:
            delete_buffer.append(json.loads(line))
            if len(delete_buffer) >= delete_batch_size:
                collection.delete(ids=delete_buffer)
                total_deleted += len(delete_buffer)
                delete_buffer.clear()
        if delete_buffer:
            collection.delete(ids=delete_buffer)
            total_deleted += len(delete_buffer)

    return total_deleted


def purge_stale_source_ids(
    collection,
    source: str,
    valid_ids: set[str],
    fetch_size: int = 5000,
    delete_batch_size: int = 1000,
) -> int:
    if fetch_size <= 0:
        raise ValueError("fetch_size must be positive")

    def stale_ids():
        offset = 0
        while True:
            result = collection.get(
                where={"source": source}, limit=fetch_size, offset=offset, include=[]
            )
            batch = result.get("ids") or []
            yield from (cid for cid in batch if cid not in valid_ids)
            offset += len(batch)
            if len(batch) < fetch_size:
                break

    return delete_ids_after_scan(collection, stale_ids(), delete_batch_size)


def purge_stale_data_streaming(
    collection,
    valid_sources: set[str],
    fetch_size: int = 5000,
    delete_batch_size: int = 1000,
    logger=None,
) -> int:
    if fetch_size <= 0 or delete_batch_size <= 0:
        raise ValueError("Cleanup batch sizes must be positive")
    total_deleted = delete_ids_after_scan(
        collection,
        iter_orphan_ids(collection, valid_sources, fetch_size),
        delete_batch_size,
    )
    if logger:
        if total_deleted:
            logger.warning("Found and deleted %s orphaned chunks.", total_deleted)
        else:
            logger.info("No orphaned data found.")
    return total_deleted
