from rag_pipeline.cleanup import (
    compute_stale_ids,
    iter_orphan_ids,
    purge_stale_data_streaming,
)


class FakeCollection:
    def __init__(self, rows):
        self._rows = rows
        self.deleted_ids = []

    def get(self, limit=5000, offset=0, include=None):
        batch = self._rows[offset : offset + limit]
        return {
            "ids": [r["id"] for r in batch],
            "metadatas": [r["meta"] for r in batch],
        }

    def delete(self, ids):
        self.deleted_ids.extend(ids)


def test_compute_stale_ids():
    stale = compute_stale_ids({"a", "b", "c"}, {"a", "c"})
    assert stale == ["b"]


def test_iter_orphan_ids_chunks():
    rows = [
        {"id": "1", "meta": {"source": "a.pdf"}},
        {"id": "2", "meta": {"source": "orphan.pdf"}},
        {"id": "3", "meta": {"source": "b.pdf"}},
        {"id": "4", "meta": {"source": "orphan.pdf"}},
    ]
    coll = FakeCollection(rows)
    orphan_ids = list(
        iter_orphan_ids(coll, valid_sources={"a.pdf", "b.pdf"}, fetch_size=2)
    )
    assert orphan_ids == ["2", "4"]


def test_purge_stale_data_streaming_deletes_in_batches():
    rows = [{"id": str(i), "meta": {"source": "orphan.pdf"}} for i in range(5)]
    coll = FakeCollection(rows)
    deleted = purge_stale_data_streaming(
        coll,
        valid_sources={"a.pdf"},
        fetch_size=2,
        delete_batch_size=2,
    )
    assert deleted == 5
    assert coll.deleted_ids == ["0", "1", "2", "3", "4"]


class MutatingCollection(FakeCollection):
    def delete(self, ids):
        super().delete(ids)
        self._rows = [r for r in self._rows if r["id"] not in ids]


def test_cleanup_does_not_skip_rows_after_real_deletions():
    rows = [
        {"id": str(i), "meta": {"source": "keep.pdf" if i % 3 == 0 else "old.pdf"}}
        for i in range(30)
    ]
    coll = MutatingCollection(rows)
    deleted = purge_stale_data_streaming(
        coll, {"keep.pdf"}, fetch_size=4, delete_batch_size=3
    )
    assert deleted == 20
    assert len(coll._rows) == 10
    assert all(r["meta"]["source"] == "keep.pdf" for r in coll._rows)


def test_missing_metadata_is_orphaned():
    class MissingMetadata:
        def get(self, **kwargs):
            return {"ids": ["a", "b"], "metadatas": None}

    assert list(iter_orphan_ids(MissingMetadata(), set(), fetch_size=10)) == ["a", "b"]
