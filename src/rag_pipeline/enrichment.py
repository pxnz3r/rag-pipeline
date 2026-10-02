from __future__ import annotations

from dataclasses import replace
from time import sleep

from .models import Chunk


def select_enriched_chunks(
    chunks: list[Chunk], enriched_map: dict[str, Chunk]
) -> list[Chunk]:
    """Reuse only current cache entries; this operation never calls an API."""
    result = []
    for chunk in chunks:
        cached = enriched_map.get(chunk.id)
        if (
            cached
            and (cached.file_hash, cached.pipeline_version, cached.text)
            == (chunk.file_hash, chunk.pipeline_version, chunk.text)
            and cached.text_with_context
        ):
            result.append(cached)
        else:
            result.append(chunk)
    return result


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
    if periodic_save_every <= 0 or sleep_seconds < 0:
        raise ValueError("Invalid enrichment save interval or sleep duration")
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
                and cached_chunk.text == chunk.text
                and bool(cached_chunk.text_with_context)
            ):
                continue
        to_process.append(chunk)

    for i, chunk in enumerate(to_process):
        try:
            context = generate_context_fn(chunk)
            if not isinstance(context, str) or not context.strip():
                raise ValueError("Empty enrichment response")
            enriched_chunk = replace(chunk)
            chunk_text = chunk.text or ""
            enriched_chunk.text_with_context = (
                f"CONTEXT: {context}\n\nORIGINAL TEXT:\n{chunk_text}"
            )
            enriched_map[chunk.id] = enriched_chunk
            dirty = True
            sleep(sleep_seconds)
        except Exception as exc:  # noqa: BLE001 - caller may inject various API exception types
            if logger:
                logger.error("Failed to enrich %s (%s)", chunk.id, type(exc).__name__)
            # Do not cache a transient API failure as a successful enrichment,
            # and do not serve a cached context from an older document version.
            enriched_map.pop(chunk.id, None)
            dirty = True

        if dirty and (i + 1) % periodic_save_every == 0:
            persist_fn(enriched_map)
            dirty = False

    if dirty:
        persist_fn(enriched_map)
    return [enriched_map.get(chunk.id, chunk) for chunk in chunks]
