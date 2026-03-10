from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pypdf
from pypdf.errors import PdfReadError

from .models import Chunk
from .storage import get_file_hash, save_json_atomic
from .text import is_text_garbled, sanitize_text


def process_pdf_page_level(
    pdf_path: Path,
    processed_state: Dict[str, Tuple[str, str]],
    *,
    pipeline_version: str,
    min_chunk_chars: int = 100,
    garble_threshold: float = 0.40,
    timeout_ctx=None,
    logger=None,
) -> list[Chunk]:
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
            return []

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
                        return []
                except (
                    PdfReadError,
                    TimeoutError,
                    ValueError,
                    OSError,
                ) as exc:
                    if logger:
                        logger.error("Error decrypting %s: %s", pdf_path.name, exc)
                    return []

            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                cnt_total += 1
                text = sanitize_text(raw_text)
                if len(text) < min_chunk_chars:
                    cnt_short += 1
                    continue

                is_garbled_flag = False
                if is_text_garbled(text, threshold=garble_threshold):
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

                chunks.append(
                    Chunk(
                        id=f"{pdf_path.name}::p{page_num}::i0",
                        text=text,
                        pdf_name=pdf_path.name,
                        file_hash=current_hash,
                        pipeline_version=pipeline_version,
                        page_number=page_num,
                        chunk_index=0,
                        pdf_path=str(pdf_path),
                        char_count=len(text),
                        word_count=len(text.split()),
                        has_numbers=any(c.isdigit() for c in text),
                        has_formula=any(c in text for c in ["$", "%", "="]),
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
        return []

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
    dirty = False
    for i, pdf in enumerate(all_pdfs):
        new_chunks = process_fn(pdf, processed_state)
        if new_chunks:
            old_ids = [
                cid
                for cid, old_chunk in chunk_map.items()
                if old_chunk.pdf_name == pdf.name
            ]
            for cid in old_ids:
                del chunk_map[cid]
            for chunk in new_chunks:
                chunk_map[chunk.id] = chunk
            dirty = True
        if i > 0 and i % gc_interval_pdfs == 0:
            gc.collect()

    if dirty:
        sorted_chunks = sorted(
            [c.to_dict() for c in chunk_map.values()], key=lambda x: x["id"]
        )
        save_json_atomic(
            sorted_chunks, processed_dir / master_chunks_file, logger=logger
        )
        if logger:
            logger.info("Updated master chunks with %s items.", len(chunk_map))
    else:
        if logger:
            logger.info("No new PDF content detected.")
    return list(chunk_map.values())
