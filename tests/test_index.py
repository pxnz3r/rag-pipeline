import json
import sqlite3

import numpy as np
import pytest

from rag_pipeline import Index


class Encoder:
    signature = "test:deterministic:v1"

    def encode(self, texts):
        return [
            [1 if "revenue" in t.lower() else 0, 1 if "lease" in t.lower() else 0, 1]
            for t in texts
        ]


def source(root, filename, text, **meta):
    root.mkdir(exist_ok=True)
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if meta:
        path.with_name(path.name + ".meta.json").write_text(json.dumps(meta))
    return path


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    source(
        root,
        "a/report.txt",
        "Revenue was USD 120 million in 2024.",
        company="A",
        year=2024,
    )
    source(
        root,
        "b/report.txt",
        "Revenue was USD 600 million in 2024.",
        company="B",
        year=2024,
    )
    return root


def test_sync_snapshot_filters_and_citations(tmp_path, corpus):
    path = tmp_path / "index.sqlite"
    with Index(path) as index:
        assert index.ingest(corpus)["changed"] == 2
        assert index.ingest(corpus)["generation"] == 1
        hits = index.search("Revenue", filters={"company": "A", "year": "2024"})
        assert [h.document for h in hits] == ["a/report.txt"]
        assert (
            hits[0].text
            == (corpus / hits[0].document).read_text()[hits[0].start : hits[0].end]
        )
        assert not index.search("Revenue", filters={"company": "A' OR 1=1--"})
        assert not index.search("Revenue", filters={"company": "Missing"})
        source(
            corpus,
            "a/report.txt",
            "Net profit was USD 10 million.",
            company="A",
            year=2024,
        )
        (corpus / "b/report.txt").unlink()
        stats = index.ingest(corpus)
        assert (stats["changed"], stats["removed"], stats["chunks"]) == (1, 1, 1)
        assert not index.search("Revenue")
        index.db.execute("INSERT INTO fts(fts, rank) VALUES('integrity-check', 1)")
    with Index(path) as reopened:
        assert reopened.status()["generation"] == 2
        assert reopened.search("profit")[0].metadata["company"] == "A"


def test_whole_ingest_rollback_and_wal_reader(tmp_path, corpus):
    path = tmp_path / "index.sqlite"
    with Index(path) as index, Index(path) as reader:
        index.ingest(corpus)
        source(corpus, "a/report.txt", "Replacement revenue USD 999 million.")
        bad = source(corpus, "z-bad.json", "{invalid json")
        before = index.status()
        with pytest.raises(ValueError):
            index.ingest(corpus)
        assert index.status() == before
        assert "120" in index.search("Revenue", filters={"company": "A"})[0].text
        bad.unlink()
        with reader._transaction():
            assert (
                reader.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
            )
            (corpus / "b/report.txt").unlink()
            index.ingest(corpus)
            assert (
                reader.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
            )
        assert reader.status()["documents"] == 1
        with pytest.raises(ValueError, match="absent"):
            index.ingest(tmp_path / "missing")
        assert index.status()["documents"] == 1


def test_dense_roundtrip_filtering_and_model_consistency(tmp_path, corpus):
    path = tmp_path / "index.sqlite"
    with Index(path, Encoder()) as index:
        index.ingest(corpus)
        assert [
            h.document
            for h in index.search(
                "lease", filters={"company": "B"}, mode="dense", min_cosine=0
            )
        ] == ["b/report.txt"]

        class Bad(Encoder):
            def encode(self, texts):
                return np.full((len(texts), 3), np.nan)

        index.embedder = Bad()
        source(corpus, "new.txt", "A new document.")
        with pytest.raises(ValueError, match="non-finite"):
            index.ingest(corpus)
        assert index.status()["documents"] == 2
        index.embedder = Encoder()
        index.embedder.signature = "different-model"
        with pytest.raises(ValueError, match="does not match"):
            index.search("Revenue")
    with Index(path) as offline:
        assert offline.search("Revenue", mode="lexical")
        with pytest.raises(ValueError, match="same embedder"):
            offline.ingest(corpus)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"candidates": 0},
        {"mode": "bogus"},
        {"context_chars": 0},
        {"min_cosine": float("nan")},
        {"filters": {"bad.key": "value"}},
    ],
)
def test_invalid_search_parameters(tmp_path, kwargs):
    with Index(tmp_path / "index.sqlite") as index:
        with pytest.raises(ValueError):
            index.search("Revenue", **kwargs)


def test_corrupt_database_and_symlink_refused(tmp_path, corpus):
    broken = tmp_path / "broken.sqlite"
    broken.write_bytes(b"not a database")
    with pytest.raises(sqlite3.DatabaseError):
        Index(broken)
    (corpus / "linked.txt").symlink_to(corpus / "a/report.txt")
    with Index(tmp_path / "index.sqlite") as index:
        with pytest.raises(ValueError, match="Symlink"):
            index.ingest(corpus)
        assert not index.status()["documents"]


def test_chunk_overlap_and_bounded_context(tmp_path):
    root = tmp_path / "corpus"
    text = "Introduction.\n" + "Revenue increases steadily.\n" * 100
    source(root, "book.md", text)
    with Index(tmp_path / "index.sqlite") as index:
        index.ingest(root, size=300, overlap=30)
        hits = index.search("Revenue", context_chars=400)
        assert hits and all(
            len(h.text) <= 400 and h.text == text[h.start : h.end] for h in hits
        )


def test_reranker_order_and_invalid_scores(tmp_path, corpus):
    class Ranker:
        def score(self, question, texts):
            return [2 if "600" in text else 1 for text in texts]

    with Index(tmp_path / "index.sqlite", reranker=Ranker()) as index:
        index.ingest(corpus)
        hits = index.search("Revenue")
        assert hits[0].document == "b/report.txt" and hits[0].score == 2
        index.reranker.score = lambda *_: [float("nan")]
        with pytest.raises(ValueError, match="reranker"):
            index.search("Revenue")
        assert index.status()["documents"] == 2


def test_empty_dense_index_and_context_smaller_than_chunk(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    with Index(tmp_path / "index.sqlite", Encoder()) as index:
        index.ingest(root)
        assert not index.search("Revenue")
        source(root, "book.txt", "Revenue is recorded. " * 100)
        index.ingest(root)
        hit = index.search("Revenue", context_chars=100)[0]
        assert len(hit.text) <= 100 and hit.end - hit.start == len(hit.text)


def test_long_book_search_does_not_copy_whole_section_per_candidate(tmp_path):
    import tracemalloc

    root = tmp_path / "corpus"
    text = "📚 Évidence: Revenue and contractual obligations.\n" * 50000
    source(root, "long-book.txt", text)
    with Index(tmp_path / "index.sqlite") as index:
        index.ingest(root)
        tracemalloc.start()
        try:
            hits = index.search("Revenue")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # Python heap only: the old per-candidate copies used ~40 section sizes.
        assert hits and peak < len(text) * 3
        assert all(h.text == text[h.start : h.end] for h in hits)


def test_missing_schema_is_not_silently_reinitialized(tmp_path, corpus):
    path = tmp_path / "index.sqlite"
    with Index(path) as index:
        index.ingest(corpus)
        index.db.execute("DELETE FROM state WHERE key='schema'")
    with pytest.raises(ValueError, match="schema"):
        Index(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert not connection.execute(
            "SELECT value FROM state WHERE key='schema'"
        ).fetchone()

    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO state VALUES('schema','2')")
        connection.execute("DROP TABLE fts")
    with pytest.raises(ValueError, match="Incomplete"):
        Index(path)
    with sqlite3.connect(path) as connection:
        assert not connection.execute(
            "SELECT name FROM sqlite_master WHERE name='fts'"
        ).fetchone()


def test_unreadable_directory_never_prunes_existing_corpus(tmp_path, corpus):
    import os

    if os.name != "posix" or os.geteuid() == 0:
        pytest.skip("Requires POSIX directory permissions for a non-root user")
    with Index(tmp_path / "index.sqlite") as index:
        index.ingest(corpus)
        before = index.status()
        corpus.chmod(0)
        try:
            with pytest.raises(PermissionError):
                index.ingest(corpus)
        finally:
            corpus.chmod(0o700)
        assert index.status() == before
