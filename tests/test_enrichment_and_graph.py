import asyncio
import json
from dataclasses import replace

import pytest

from rag_pipeline.bm25_utils import build_bm25_index
from rag_pipeline.enrichment import batch_enrich_chunks
from rag_pipeline.lightrag_pipeline import sync_lightrag
from rag_pipeline.models import Chunk


def chunk(text="source text"):
    return Chunk("a", text, "a.pdf", "hash", "2.0", 1, 0)


def test_transient_enrichment_failure_retries_and_does_not_mutate_master():
    c = chunk()
    cache = {}
    saves = []

    def failure(c):
        raise RuntimeError("offline")

    result = batch_enrich_chunks(
        [c],
        enriched_map=cache,
        generate_context_fn=failure,
        persist_fn=lambda data: saves.append(dict(data)),
        sleep_seconds=0,
    )
    assert result == [c]
    assert cache == {}
    assert c.text_with_context is None
    result = batch_enrich_chunks(
        [c],
        enriched_map=cache,
        generate_context_fn=lambda c: "context",
        persist_fn=lambda data: None,
        sleep_seconds=0,
    )
    assert "CONTEXT: context" in result[0].text_with_context
    assert c.text_with_context is None


def test_enrichment_content_change_invalidates_cache():
    old = chunk("old")
    old.text_with_context = "old context"
    cache = {"a": old}
    result = batch_enrich_chunks(
        [chunk("new")],
        enriched_map=cache,
        generate_context_fn=lambda c: "new context",
        persist_fn=lambda data: None,
        sleep_seconds=0,
    )
    assert result[0].text_with_context.endswith("new")


def test_bm25_skips_empty_token_documents_and_preserves_ids():
    assert build_bm25_index([chunk("!")]) == (None, {})
    index, mapping = build_bm25_index([chunk("!"), replace(chunk(), id="b")])
    assert mapping == {0: "b"}
    assert len(index.get_scores(["source"])) == 1


class Graph:
    def __init__(self, path, fail=False):
        self.path = path
        self.fail = fail
        self.inserts = []
        self.initialized = 0
        self.finalized = 0

    async def initialize_storages(self):
        self.initialized += 1

    async def finalize_storages(self):
        self.finalized += 1

    async def ainsert(self, text):
        if self.fail:
            raise RuntimeError("insert failed")
        self.inserts.append(text)


def test_graph_rebuild_reuse_and_failed_publication(tmp_path):
    async def run():
        first = await sync_lightrag([chunk()], index_dir=tmp_path, rag_factory=Graph)
        pointer = json.loads((tmp_path / "current.json").read_text())
        reused = await sync_lightrag([chunk()], index_dir=tmp_path, rag_factory=Graph)
        assert reused.path == first.path
        assert reused.inserts == []
        with pytest.raises(RuntimeError):
            await sync_lightrag(
                [chunk("changed")],
                index_dir=tmp_path,
                rag_factory=lambda p: Graph(p, fail=True),
            )
        assert json.loads((tmp_path / "current.json").read_text()) == pointer
        fresh = await sync_lightrag(
            [chunk("changed")], index_dir=tmp_path, rag_factory=Graph
        )
        assert fresh.path != first.path
        assert first.path.exists()
        empty = await sync_lightrag([], index_dir=tmp_path, rag_factory=Graph)
        assert empty.path != fresh.path
        assert empty.inserts == []

    asyncio.run(run())


def test_graph_batches_are_bounded(tmp_path):
    async def run():
        graph = await sync_lightrag(
            [chunk("x" * 55)], index_dir=tmp_path, rag_factory=Graph, batch_max_chars=10
        )
        assert all(len(text) <= 10 for text in graph.inserts)
        assert "".join(graph.inserts) == "x" * 55

    asyncio.run(run())


def test_graph_detects_silently_failed_upstream_insert(tmp_path):
    class Status:
        async def get_status_counts(self):
            return {"processed": 1, "failed": 1}

    class SilentFailure(Graph):
        doc_status = Status()

    with pytest.raises(RuntimeError, match="incomplete"):
        asyncio.run(
            sync_lightrag([chunk()], index_dir=tmp_path, rag_factory=SilentFailure)
        )
    assert not (tmp_path / "current.json").exists()


def test_corrupt_graph_pointer_is_not_overwritten(tmp_path):
    pointer = tmp_path / "current.json"
    pointer.write_text('{"generation":"../../escape","fingerprint":"bad"}')
    before = pointer.read_bytes()
    with pytest.raises(ValueError, match="pointer"):
        asyncio.run(sync_lightrag([chunk()], index_dir=tmp_path, rag_factory=Graph))
    assert pointer.read_bytes() == before


def test_offline_cache_selection_rejects_stale_and_deleted_entries():
    from rag_pipeline.enrichment import select_enriched_chunks

    c = chunk("current")
    stale = replace(c, text="old", text_with_context="stale context")
    removed = replace(c, id="removed")
    assert select_enriched_chunks([c], {c.id: stale, "removed": removed}) == [c]
    fresh = replace(c, text_with_context="fresh context")
    assert select_enriched_chunks([c], {c.id: fresh}) == [fresh]
