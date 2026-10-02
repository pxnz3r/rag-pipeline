import json
from concurrent.futures import ThreadPoolExecutor

import pypdf
import pytest

from rag_pipeline.config import Settings, load_settings
from rag_pipeline.ingestion import ingest_pdfs
from rag_pipeline.manifest import load_manifest, update_manifest
from rag_pipeline.models import Chunk
from rag_pipeline.processing import (
    PDFProcessingError,
    split_page_text,
    sync_master_chunks,
)
from rag_pipeline.storage import load_chunks_map, save_json_atomic


def chunk(name="a.pdf", text="old"):
    return Chunk(f"{name}::p1::i0", text, name, "hash", "2.0", 1, 0)


def test_atomic_failure_preserves_previous_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.json"
    save_json_atomic({"old": True}, path)
    with pytest.raises(TypeError):
        save_json_atomic({"bad": object()}, path)
    assert json.loads(path.read_text()) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_writers_do_not_share_temporary_files(tmp_path):
    path = tmp_path / "checkpoint.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda i: save_json_atomic({"writer": i, "rows": [i] * 100}, path),
                range(40),
            )
        )
    data = json.loads(path.read_text())
    assert data["rows"] == [data["writer"]] * 100


@pytest.mark.parametrize("data", [{}, [None], [{"id": "a"}], [chunk().to_dict()] * 2])
def test_invalid_checkpoint_fails_closed(tmp_path, data):
    save_json_atomic(data, tmp_path / "c.json")
    with pytest.raises(ValueError):
        load_chunks_map(tmp_path, "c.json", 10000)


def test_missing_checkpoint_is_empty_but_oversize_is_error(tmp_path):
    assert load_chunks_map(tmp_path, "missing.json", 100) == {}
    save_json_atomic([chunk().to_dict()], tmp_path / "c.json")
    with pytest.raises(ValueError):
        load_chunks_map(tmp_path, "c.json", 10)


def test_corrupt_manifest_is_never_overwritten(tmp_path):
    path = tmp_path / "manifest_chroma.json"
    path.write_text('{"bad":')
    with pytest.raises(ValueError):
        update_manifest(tmp_path, "chroma", "a.pdf", "hash")
    assert path.read_text() == '{"bad":'


def test_manifest_concurrent_updates_and_invalid_store(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda i: update_manifest(tmp_path, "chroma", str(i), str(i)), range(20)
            )
        )
    assert len(load_manifest(tmp_path, "chroma")) == 20
    with pytest.raises(ValueError):
        load_manifest(tmp_path, "../../escape")


def test_master_sync_removes_deleted_and_successfully_empty_documents(tmp_path):
    chunks = {c.id: c for c in [chunk(), chunk("deleted.pdf")]}
    result = sync_master_chunks(
        [tmp_path / "a.pdf"],
        chunk_map=chunks,
        processed_state={},
        process_fn=lambda *args: [],
        gc_interval_pdfs=50,
        processed_dir=tmp_path,
        master_chunks_file="master.json",
    )
    assert result == []
    assert chunks == {}
    assert json.loads((tmp_path / "master.json").read_text()) == []


def test_master_sync_failure_preserves_memory_and_disk(tmp_path):
    chunks = {c.id: c for c in [chunk(), chunk("deleted.pdf")]}
    save_json_atomic([c.to_dict() for c in chunks.values()], tmp_path / "master.json")
    before = (tmp_path / "master.json").read_bytes()

    def fail(*args):
        raise PDFProcessingError("broken PDF")

    with pytest.raises(PDFProcessingError):
        sync_master_chunks(
            [tmp_path / "a.pdf"],
            chunk_map=chunks,
            processed_state={},
            process_fn=fail,
            gc_interval_pdfs=50,
            processed_dir=tmp_path,
            master_chunks_file="master.json",
        )
    assert len(chunks) == 2
    assert (tmp_path / "master.json").read_bytes() == before


def test_page_chunks_bound_and_cover_input():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = split_page_text(text, max_chars=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    assert all(word in " ".join(chunks) for word in text.split())
    assert split_page_text("x" * 1000, 400, 50)[-1].endswith("x")


def test_local_ingestion_blank_and_corrupt_pdf(tmp_path):
    settings = Settings(tmp_path)
    settings.ensure_directories()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(settings.data_dir / "blank.pdf")
    assert ingest_pdfs(settings) == []
    (settings.data_dir / "bad.pdf").write_bytes(b"not a PDF")
    with pytest.raises(PDFProcessingError):
        ingest_pdfs(settings)


@pytest.mark.parametrize(
    "name,value",
    [
        ("GROQ_TIMEOUT", "nan"),
        ("RETRIEVAL_TOP_K", "0"),
        ("RERANK_BATCH_SIZE", "oops"),
        ("CHUNK_OVERLAP_CHARS", "9999"),
    ],
)
def test_bad_settings_rejected_before_making_directories(
    tmp_path, monkeypatch, name, value
):
    monkeypatch.setenv(name, value)
    root = tmp_path / "new"
    with pytest.raises(ValueError):
        load_settings(root)
    assert not root.exists()


def write_text_pdf(path, text):
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 10 10 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


def test_real_pdf_chunking_incremental_ingestion_and_config_invalidation(tmp_path):
    from dataclasses import replace

    settings = Settings(tmp_path, chunk_max_chars=300, chunk_overlap_chars=30)
    settings.ensure_directories()
    write_text_pdf(
        settings.data_dir / "book.pdf", "Risk management and position sizing. " * 50
    )
    first = ingest_pdfs(settings)
    assert len(first) > 1
    assert all(len(c.text) <= 300 for c in first)
    checkpoint = settings.processed_dir / "master_chunks.json"
    before = checkpoint.stat().st_mtime_ns
    assert ingest_pdfs(settings) == first
    assert checkpoint.stat().st_mtime_ns == before
    second = ingest_pdfs(replace(settings, chunk_max_chars=500))
    assert len(second) < len(first)
    assert second[0].pipeline_version != first[0].pipeline_version
    (settings.data_dir / "book.pdf").unlink()
    assert ingest_pdfs(settings) == []


def test_master_sync_failed_commit_does_not_mutate_memory(tmp_path, monkeypatch):
    import rag_pipeline.processing as processing

    c = chunk()
    current = {c.id: c}

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(processing, "save_json_atomic", fail)
    with pytest.raises(OSError):
        sync_master_chunks(
            [tmp_path / "a.pdf"],
            chunk_map=current,
            processed_state={},
            process_fn=lambda *args: [],
            gc_interval_pdfs=50,
            processed_dir=tmp_path,
            master_chunks_file="master.json",
        )
    assert current == {c.id: c}
