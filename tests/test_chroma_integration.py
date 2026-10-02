"""Real persistent Chroma tests with local synthetic embeddings and no downloads."""

import pytest

from rag_pipeline.chroma_pipeline import populate_chromadb
from rag_pipeline.cleanup import purge_stale_data_streaming
from rag_pipeline.models import Chunk

chromadb = pytest.importorskip("chromadb")
EmbeddingFunction = pytest.importorskip("chromadb.api.types").EmbeddingFunction


class TinyEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return TinyEmbedding()

    def __call__(self, input):
        return [[float(len(text)), 1.0] for text in input]

    @staticmethod
    def name():
        return "local-test-embedding"


def test_real_persistent_store_sync_and_cleanup(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection(
        "regression", embedding_function=TinyEmbedding()
    )
    manifest = {}

    def sync(chunks):
        return populate_chromadb(
            chunks=chunks,
            collection=collection,
            manifest=manifest,
            batch_size=2,
            update_manifest_bulk_fn=lambda store, updates: manifest.update(updates),
            remove_manifest_entries_fn=lambda store, names: [
                manifest.pop(n, None) for n in names
            ],
        )

    chunks = [
        Chunk(f"a{i}", f"text {i}", "a.pdf", "hash", "2.0", i + 1, 0) for i in range(7)
    ]
    sync(chunks)
    assert collection.count() == 7
    sync(chunks[:2])
    assert collection.count() == 2
    result = collection.get(ids=["a0"], include=["metadatas"])
    assert result["metadatas"][0]["pipeline_version"] == "2.0"
    collection.upsert(
        ids=[f"orphan{i}" for i in range(13)],
        documents=["stale"] * 13,
        metadatas=[{"source": "gone.pdf"}] * 13,
    )
    assert (
        purge_stale_data_streaming(
            collection, {"a.pdf"}, fetch_size=3, delete_batch_size=2
        )
        == 13
    )
    assert collection.count() == 2
    sync([])
    assert collection.count() == 0
    assert manifest == {}
