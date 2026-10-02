from __future__ import annotations

import inspect
import json
import re
import uuid
from pathlib import Path

from .chroma_pipeline import document_fingerprint
from .manifest import file_lock
from .models import Chunk
from .storage import save_json_atomic


async def _close(rag) -> None:
    finalize = getattr(rag, "finalize_storages", None)
    if finalize:
        result = finalize()
        if inspect.isawaitable(result):
            await result


async def _flush_and_verify(rag) -> None:
    # ainsert() can catch errors and mark documents FAILED rather than raise.
    status = getattr(rag, "doc_status", None)
    if status is not None and hasattr(status, "get_status_counts"):
        counts = await status.get_status_counts()
        if any(
            count
            for state, count in counts.items()
            if str(getattr(state, "value", state)).lower() != "processed"
        ):
            raise RuntimeError("Graph build contains incomplete or failed documents")
    for name in (
        "full_docs",
        "text_chunks",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb",
        "chunk_entity_relation_graph",
        "llm_response_cache",
        "doc_status",
    ):
        storage = getattr(rag, name, None)
        if storage is not None and hasattr(storage, "index_done_callback"):
            await storage.index_done_callback()


async def sync_lightrag(
    chunks: list[Chunk],
    *,
    index_dir: Path,
    rag_factory,
    index_signature: str = "default",
    force_rebuild: bool = False,
    batch_max_chars: int = 12000,
):
    """Build a complete graph generation before publishing its pointer.

    Append-only graph APIs cannot reliably remove relationships derived from old
    PDFs. A changed corpus gets a fresh generation. Failed builds never replace
    the active generation; previous generations remain available for rollback.
    The caller owns the returned instance and must finalize it after use.
    """
    if batch_max_chars <= 0:
        raise ValueError("batch_max_chars must be positive")
    if len({c.id for c in chunks}) != len(chunks):
        raise ValueError("Duplicate graph chunk IDs")
    index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    pointer = index_dir / "current.json"
    fingerprint = document_fingerprint(chunks, index_signature)
    # Fail immediately if another ingestion is active; never block an event loop.
    with file_lock(pointer, timeout=0):
        current = None
        if pointer.exists():
            current = json.loads(pointer.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                raise ValueError("Invalid graph generation pointer")
            generation = current.get("generation")
            fingerprint_value = current.get("fingerprint")
            if (
                not isinstance(generation, str)
                or not re.fullmatch(r"[a-f0-9]{32}", generation)
                or not isinstance(fingerprint_value, str)
                or not re.fullmatch(r"v2:[a-f0-9]{64}", fingerprint_value)
            ):
                raise ValueError("Invalid graph generation pointer")
        reuse = (
            current and current.get("fingerprint") == fingerprint and not force_rebuild
        )
        generation = current["generation"] if reuse else uuid.uuid4().hex
        work_dir = index_dir / "generations" / generation
        if reuse and not work_dir.is_dir():
            raise ValueError("Active graph generation is missing")
        work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        rag = rag_factory(work_dir)
        if inspect.isawaitable(rag):
            rag = await rag
        try:
            initialize = getattr(rag, "initialize_storages", None)
            if initialize:
                await initialize()
            if reuse:
                return rag
            # Stable order makes rebuilds reproducible and preserves page narrative.
            buffer, size = [], 0
            for chunk in sorted(
                chunks, key=lambda c: (c.pdf_name, c.page_number, c.chunk_index)
            ):
                # Bound calls even when receiving older page-sized checkpoints.
                for start in range(0, len(chunk.text), batch_max_chars):
                    text = chunk.text[start : start + batch_max_chars]
                    if buffer and size + len(text) + 1 > batch_max_chars:
                        await rag.ainsert("\n".join(buffer))
                        buffer, size = [], 0
                    buffer.append(text)
                    size += len(text) + 1
            if buffer:
                await rag.ainsert("\n".join(buffer))
            # Keep the initialized instance usable; finalization is not reversible
            # in modern LightRAG and can also suppress individual store failures.
            await _flush_and_verify(rag)
            save_json_atomic(
                {"generation": generation, "fingerprint": fingerprint}, pointer
            )
            return rag
        except BaseException:
            await _close(rag)
            raise
