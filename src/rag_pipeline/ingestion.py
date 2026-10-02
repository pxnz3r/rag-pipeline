from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from functools import partial

from .config import Settings
from .manifest import file_lock
from .processing import process_pdf_page_level, sync_master_chunks
from .storage import load_chunks_map


def ingest_pdfs(settings: Settings):
    """Local PDF-to-checkpoint ingestion without network, models, or API keys."""
    settings.ensure_directories()
    extraction_signature = hashlib.sha256(
        json.dumps(
            [
                settings.min_chunk_chars,
                settings.chunk_max_chars,
                settings.chunk_overlap_chars,
                settings.max_pdf_bytes,
                settings.max_pdf_pages,
            ]
        ).encode()
    ).hexdigest()[:12]
    version = f"{settings.pipeline_version}:{extraction_signature}"
    filename = "master_chunks.json"
    with file_lock(settings.processed_dir / filename):
        chunks = load_chunks_map(
            settings.processed_dir, filename, settings.max_checkpoint_bytes
        )
        states = defaultdict(set)
        for chunk in chunks.values():
            states[chunk.pdf_name].add((chunk.file_hash, chunk.pipeline_version))
        processed = {
            name: next(iter(versions))
            for name, versions in states.items()
            if len(versions) == 1
        }
        return sync_master_chunks(
            sorted(settings.data_dir.glob("*.pdf")),
            chunk_map=chunks,
            processed_state=processed,
            process_fn=partial(
                process_pdf_page_level,
                pipeline_version=version,
                min_chunk_chars=settings.min_chunk_chars,
                chunk_max_chars=settings.chunk_max_chars,
                chunk_overlap_chars=settings.chunk_overlap_chars,
                max_pdf_bytes=settings.max_pdf_bytes,
                max_pdf_pages=settings.max_pdf_pages,
            ),
            gc_interval_pdfs=settings.gc_interval_pdfs,
            processed_dir=settings.processed_dir,
            master_chunks_file=filename,
        )
