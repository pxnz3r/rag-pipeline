from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pypdf
from pypdf.errors import PdfReadError

from .models import Chunk
from .storage import get_file_hash, save_json_atomic
from .text import is_text_garbled, sanitize_text


class PDFProcessingError(RuntimeError):
    """A PDF could not be read completely; its previous checkpoint is preserved."""


def split_page_text(text: str, max_chars: int = 1800, overlap: int = 200) -> list[str]:
    """Bound embedding input while retaining overlap and stable page-local IDs."""
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("Invalid chunk size or overlap")
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start + overlap:
                end = boundary
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end == len(text):
            break
        start = end - overlap
    return parts


def process_pdf_page_level(
    pdf_path: Path,
    processed_state: Dict[str, Tuple[str, str]],
    *,
    pipeline_version: str,
    min_chunk_chars: int = 100,
    garble_threshold: float = 0.40,
    timeout_ctx=None,
    logger=None,
    chunk_max_chars: int = 1800,
    chunk_overlap_chars: int = 200,
    max_pdf_bytes: int = 100 * 1024 * 1024,
    max_pdf_pages: int = 10000,
) -> list[Chunk] | None:
    if min_chunk_chars <= 0 or max_pdf_bytes <= 0 or max_pdf_pages <= 0:
        raise ValueError("PDF processing limits must be positive")
    if not 0 <= garble_threshold <= 1:
        raise ValueError("garble_threshold must be between 0 and 1")
    split_page_text("", chunk_max_chars, chunk_overlap_chars)
    if pdf_path.stat().st_size > max_pdf_bytes:
        raise PDFProcessingError(f"PDF exceeds byte limit: {pdf_path.name}")
    current_hash = get_file_hash(pdf_path, logger=logger)
    if not current_hash:
        if logger:
            logger.warning(
                "Could not calculate hash for %s. Reprocessing to be safe.",
                pdf_path.name,
            )
    elif pdf_path.name in processed_state:
        stored_hash, stored_ver = processed_state[pdf_path.name]
        if stored_hash == current_hash and stored_ver == pipeline_version:
            return None

    chunks: list[Chunk] = []
    cnt_total = cnt_kept = cnt_garbled = cnt_short = cnt_kept_garbled = 0

    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            if reader.is_encrypted:
                try:
                    if timeout_ctx:
                        with timeout_ctx(5):
                            decrypt_result = reader.decrypt("")
                    else:
                        decrypt_result = reader.decrypt("")
                    if decrypt_result == 0:
                        if logger:
                            logger.warning(
                                "Skipping encrypted PDF (password required): %s",
                                pdf_path.name,
                            )
                        raise PDFProcessingError(f"Password required: {pdf_path.name}")
                except (
                    PdfReadError,
                    TimeoutError,
                    ValueError,
                    OSError,
                ) as exc:
                    if logger:
                        logger.error("Error decrypting %s: %s", pdf_path.name, exc)
                    raise PDFProcessingError(
                        f"Cannot decrypt: {pdf_path.name}"
                    ) from exc

            if len(reader.pages) > max_pdf_pages:
                raise PDFProcessingError(f"PDF exceeds page limit: {pdf_path.name}")
            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                cnt_total += 1
                text = sanitize_text(raw_text)
                if len(text) < min_chunk_chars:
                    cnt_short += 1
                    continue

                is_garbled_flag = False
                if is_text_garbled(raw_text, threshold=garble_threshold):
                    if len(text) > 2000:
                        is_garbled_flag = True
                        cnt_kept_garbled += 1
                        if logger:
                            logger.warning(
                                "Keeping garbled but large page %s in %s (%s chars)",
                                page_num,
                                pdf_path.name,
                                len(text),
                            )
                    else:
                        cnt_garbled += 1
                        continue

                for chunk_index, chunk_text in enumerate(
                    split_page_text(text, chunk_max_chars, chunk_overlap_chars)
                ):
                    chunks.append(
                        Chunk(
                            id=f"{pdf_path.name}::p{page_num}::i{chunk_index}",
                            text=chunk_text,
                            pdf_name=pdf_path.name,
                            file_hash=current_hash,
                            pipeline_version=pipeline_version,
                            page_number=page_num,
                            chunk_index=chunk_index,
                            pdf_path=str(pdf_path),
                            char_count=len(chunk_text),
                            word_count=len(chunk_text.split()),
                            has_numbers=any(c.isdigit() for c in chunk_text),
                            has_formula=any(c in chunk_text for c in ["$", "%", "="]),
                            is_garbled=is_garbled_flag,
                        )
                    )
                cnt_kept += 1
    except (
        OSError,
        PdfReadError,
        UnicodeError,
        ValueError,
        TimeoutError,
    ) as exc:
        if logger:
            logger.error("Error processing %s: %s", pdf_path, exc)
        raise PDFProcessingError(f"Could not process PDF: {pdf_path.name}") from exc

    if cnt_total > 0 and logger:
        logger.info(
            "PDF %s: %s/%s kept. (Dropped Garbled: %s, Kept Garbled: %s, Short: %s)",
            pdf_path.name,
            cnt_kept,
            cnt_total,
            cnt_garbled,
            cnt_kept_garbled,
            cnt_short,
        )
        if cnt_garbled / cnt_total > 0.5:
            logger.warning(
                "High garble rate detected for %s! Check PDF source.", pdf_path.name
            )
    return chunks


def sync_master_chunks(
    all_pdfs: Iterable[Path],
    *,
    chunk_map: Dict[str, Chunk],
    processed_state: Dict[str, tuple[str, str]],
    process_fn,
    gc_interval_pdfs: int,
    processed_dir: Path,
    master_chunks_file: str,
    logger=None,
) -> list[Chunk]:
    if gc_interval_pdfs <= 0:
        raise ValueError("gc_interval_pdfs must be positive")
    pdfs = list(all_pdfs)
    names = {pdf.name for pdf in pdfs}
    if len(names) != len(pdfs):
        raise ValueError("PDF basenames must be unique within a corpus")
    # Stage changes and commit before mutating the caller's in-memory checkpoint.
    staged_map = {cid: c for cid, c in chunk_map.items() if c.pdf_name in names}
    ids_by_pdf = defaultdict(set)
    for cid, chunk in staged_map.items():
        ids_by_pdf[chunk.pdf_name].add(cid)
    dirty = len(staged_map) != len(chunk_map)
    for i, pdf in enumerate(pdfs):
        new_chunks = process_fn(pdf, processed_state)
        if new_chunks is not None:
            old_ids = ids_by_pdf.pop(pdf.name, set())
            for cid in old_ids:
                del staged_map[cid]
            for chunk in new_chunks:
                staged_map[chunk.id] = chunk
                ids_by_pdf[chunk.pdf_name].add(chunk.id)
            dirty = True
        if i > 0 and i % gc_interval_pdfs == 0:
            gc.collect()

    if dirty:
        sorted_chunks = sorted(
            [c.to_dict() for c in staged_map.values()], key=lambda x: x["id"]
        )
        save_json_atomic(
            sorted_chunks, processed_dir / master_chunks_file, logger=logger
        )
        chunk_map.clear()
        chunk_map.update(staged_map)
        if logger:
            logger.info("Updated master chunks with %s items.", len(chunk_map))
    else:
        if logger:
            logger.info("No new PDF content detected.")
    return list(chunk_map.values())
