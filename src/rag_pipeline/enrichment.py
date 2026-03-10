from __future__ import annotations

from time import sleep

from .models import Chunk


def batch_enrich_chunks(
    chunks: list[Chunk],
    *,
    enriched_map: dict[str, Chunk],
    generate_context_fn,
    persist_fn,
    logger=None,
    sleep_seconds: float = 0.2,
    periodic_save_every: int = 20,
) -> list[Chunk]:
    valid_chunk_ids = {c.id for c in chunks}
    stale_cached_ids = [
        cid for cid in list(enriched_map.keys()) if cid not in valid_chunk_ids
    ]
    if stale_cached_ids:
        for cid in stale_cached_ids:
            enriched_map.pop(cid, None)
        if logger:
            logger.info(
                "Pruned %s stale enrichment cache entries.", len(stale_cached_ids)
            )

    dirty = bool(stale_cached_ids)
    to_process = []
    for chunk in chunks:
        if chunk.id in enriched_map:
            cached_chunk = enriched_map[chunk.id]
            if (
                cached_chunk.file_hash == chunk.file_hash
                and cached_chunk.pipeline_version == chunk.pipeline_version
            ):
                continue
        to_process.append(chunk)

    for i, chunk in enumerate(to_process):
        try:
            context = generate_context_fn(chunk)
            chunk_text = chunk.text or ""
            chunk.text_with_context = (
                f"CONTEXT: {context}\n\nORIGINAL TEXT:\n{chunk_text}"
            )
            enriched_map[chunk.id] = chunk
            dirty = True
            sleep(sleep_seconds)
        except Exception as exc:  # noqa: BLE001 - caller may inject various API exception types
            if logger:
                logger.error("Failed to enrich %s: %s", chunk.id, exc)
            chunk_text = chunk.text or ""
            chunk.text_with_context = chunk_text
            enriched_map[chunk.id] = chunk
            dirty = True

        if dirty and (i + 1) % periodic_save_every == 0:
            persist_fn(enriched_map)
            dirty = False

    if dirty:
        persist_fn(enriched_map)
    return list(enriched_map.values())
