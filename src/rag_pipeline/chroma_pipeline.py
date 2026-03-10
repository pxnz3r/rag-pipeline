from __future__ import annotations

from typing import Callable, Dict, List

from .cleanup import purge_stale_data_streaming
from .models import Chunk
from .retrieval import first_hash_by_pdf, group_chunk_ids_by_pdf


def populate_chromadb(
    *,
    chunks: List[Chunk],
    collection,
    manifest: Dict[str, str],
    update_manifest_bulk_fn: Callable[[str, Dict[str, str]], None],
    remove_manifest_entries_fn: Callable[[str, List[str]], None],
    logger=None,
    batch_size: int = 50,
) -> object:
    chunks_to_ingest: List[Chunk] = []
    pdfs_to_update = set()
    for chunk in chunks:
        if (
            chunk.pdf_name not in manifest
            or manifest[chunk.pdf_name] != chunk.file_hash
        ):
            chunks_to_ingest.append(chunk)
            pdfs_to_update.add(chunk.pdf_name)
    if not chunks_to_ingest:
        return collection

    new_ids_by_pdf = group_chunk_ids_by_pdf(chunks_to_ingest)
    for pdf_name in pdfs_to_update:
        new_ids_for_pdf = new_ids_by_pdf.get(pdf_name, set())
        if not new_ids_for_pdf:
            if logger:
                logger.error(
                    "Skipping stale deletion for %s: No new chunks found. Manual check required.",
                    pdf_name,
                )
            continue
        existing_data = collection.get(where={"source": pdf_name}, include=[])
        if existing_data and existing_data.get("ids"):
            stale_ids = list(set(existing_data.get("ids", [])) - new_ids_for_pdf)
            if stale_ids:
                collection.delete(ids=stale_ids)

    current_pdfs = {c.pdf_name for c in chunks}
    removed_pdfs = sorted(set(manifest.keys()) - current_pdfs)
    for pdf_name in removed_pdfs:
        existing_data = collection.get(where={"source": pdf_name}, include=[])
        stale_ids = existing_data.get("ids", []) if existing_data else []
        if stale_ids:
            collection.delete(ids=stale_ids)
    remove_manifest_entries_fn("chroma", removed_pdfs)

    failed_pdfs = set()
    for i in range(0, len(chunks_to_ingest), batch_size):
        batch = chunks_to_ingest[i : i + batch_size]
        ids = [c.id for c in batch]
        docs = [c.text_with_context if c.text_with_context else c.text for c in batch]
        metas = [
            {
                "source": str(c.pdf_name),
                "page": int(c.page_number),
                "is_garbled": bool(c.is_garbled),
            }
            for c in batch
        ]
        try:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
        except (ValueError, RuntimeError, OSError):
            for chunk in batch:
                failed_pdfs.add(chunk.pdf_name)

    pdf_hash_by_name = first_hash_by_pdf(chunks_to_ingest)
    manifest_updates = {}
    for pdf in pdfs_to_update:
        if pdf in failed_pdfs:
            continue
        current_hash = pdf_hash_by_name.get(pdf)
        if current_hash:
            manifest_updates[pdf] = current_hash
    update_manifest_bulk_fn("chroma", manifest_updates)
    return collection


def purge_stale_data(collection, *, valid_sources: set[str], logger=None) -> int:
    return purge_stale_data_streaming(
        collection,
        valid_sources=valid_sources,
        fetch_size=5000,
        delete_batch_size=1000,
        logger=logger,
    )
