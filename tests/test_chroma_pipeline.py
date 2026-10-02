import pytest

from rag_pipeline.chroma_pipeline import document_fingerprint, populate_chromadb
from rag_pipeline.models import Chunk


class MutableCollection:
    def __init__(self, fail=False):
        self.rows = {
            "old": ("old content", {"source": "a.pdf"}),
            "removed": ("deleted content", {"source": "removed.pdf"}),
        }
        self.fail = fail
        self.events = []

    def get(self, where=None, include=None, limit=5000, offset=0, **kwargs):
        ids = [
            cid
            for cid, (_, meta) in self.rows.items()
            if where is None or meta["source"] == where["source"]
        ]
        return {"ids": ids[offset : offset + limit]}

    def upsert(self, ids, documents, metadatas):
        self.events.append("upsert")
        if self.fail:
            raise RuntimeError("backend unavailable")
        for cid, doc, meta in zip(ids, documents, metadatas):
            self.rows[cid] = (doc, meta)

    def delete(self, ids=None, where=None):
        self.events.append("delete")
        for cid in list(self.rows):
            if (ids is not None and cid in ids) or (
                where and self.rows[cid][1]["source"] == where["source"]
            ):
                del self.rows[cid]


def sync(chunks, coll, manifest):
    return populate_chromadb(
        chunks=chunks,
        collection=coll,
        manifest=manifest,
        update_manifest_bulk_fn=lambda store, updates: manifest.update(updates),
        remove_manifest_entries_fn=lambda store, names: [
            manifest.pop(n, None) for n in names
        ],
        batch_size=1,
    )


def make_chunk(text="new", version="2.0"):
    return Chunk("new", text, "a.pdf", "hash", version, 1, 0)


def test_unchanged_ingestion_still_removes_deleted_pdf():
    c = make_chunk()
    coll = MutableCollection()
    coll.rows = {
        "new": (c.text, {"source": "a.pdf"}),
        "removed": ("old", {"source": "removed.pdf"}),
    }
    manifest = {"a.pdf": document_fingerprint([c]), "removed.pdf": "hash"}
    sync([c], coll, manifest)
    assert list(coll.rows) == ["new"]
    assert "removed.pdf" not in manifest
    assert "upsert" not in coll.events


def test_empty_corpus_cleans_collection():
    coll = MutableCollection()
    manifest = {"a.pdf": "oldhash", "removed.pdf": "oldhash"}
    sync([], coll, manifest)
    assert coll.rows == {}
    assert manifest == {}


def test_failed_upsert_preserves_old_rows_and_manifest():
    coll = MutableCollection(fail=True)
    manifest = {"a.pdf": "oldhash"}
    with pytest.raises(RuntimeError):
        sync([make_chunk()], coll, manifest)
    assert "old" in coll.rows
    assert coll.events == ["upsert"]
    assert manifest == {"a.pdf": "oldhash"}


def test_successful_upsert_precedes_stale_delete_and_version_invalidation():
    coll = MutableCollection()
    manifest = {"a.pdf": "oldhash", "removed.pdf": "oldhash"}
    sync([make_chunk()], coll, manifest)
    assert coll.events[0] == "upsert"
    assert list(coll.rows) == ["new"]
    old_fingerprint = manifest["a.pdf"]
    sync([make_chunk(version="3.0")], coll, manifest)
    assert manifest["a.pdf"] != old_fingerprint
    old_fingerprint = manifest["a.pdf"]
    sync([make_chunk(text="changed enrichment", version="3.0")], coll, manifest)
    assert manifest["a.pdf"] != old_fingerprint
