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
