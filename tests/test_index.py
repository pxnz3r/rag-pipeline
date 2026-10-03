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


def test_open_reader_does_not_wait_for_active_wal_writer(corpus, tmp_path):
    path = tmp_path / "concurrent.sqlite"
    with Index(path) as writer:
        writer.ingest(corpus)
        before = writer.status()["generation"]
        with writer._transaction(write=True):
            writer.db.execute(
                "UPDATE state SET value=? WHERE key='generation'", (str(before + 1),)
            )
            with Index(path) as reader:
                assert reader.status()["generation"] == before
                assert reader.search("revenue", mode="lexical")
        with Index(path) as reader:
            assert reader.status()["generation"] == before + 1


def test_distinct_document_selection_is_explicit_and_preserves_scope(tmp_path):
    root = tmp_path / "corpus"
    source(root, "a.txt", "revenue " * 80, company="A")
    source(root, "b.txt", "revenue " * 80, company="B")
    with Index(tmp_path / "i.sqlite") as index:
        index.ingest(root, size=100, overlap=0)
        windows = index.search("revenue", mode="lexical", k=4, diversify=False)
        assert len(windows) == 4
        distinct = index.search("revenue", mode="lexical", k=4, distinct_documents=True)
        assert {h.document for h in distinct} == {"a.txt", "b.txt"} and len(
            distinct
        ) == 2
        scoped = index.search(
            "revenue",
            mode="lexical",
            k=4,
            distinct_documents=True,
            filters={"company": "A"},
        )
        assert len(scoped) == 1 and scoped[0].document == "a.txt"


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
        import os
        import stat

        if os.name == "posix":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert (
                stat.S_IMODE(path.with_name(path.name + "-wal").stat().st_mode) == 0o600
            )
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
        {"k": 1.5},
        {"context_chars": True},
        {"candidates": 0},
        {"mode": "bogus"},
        {"mode": "dense"},
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


def test_software_symbols_stop_words_and_exact_offsets(tmp_path):
    root = tmp_path / "corpus"
    for name, text in [
        ("cpp.txt", "C++ compiler"),
        ("cs.txt", "C# compiler"),
        ("it.txt", "IT policy"),
        ("or.sql", "OR operator"),
        ("func.py", "def parse_json(): pass"),
    ]:
        source(root, name, text)
    with Index(tmp_path / "index.sqlite") as index:
        index.ingest(root)
        for query, expected in [
            ("C++", "cpp.txt"),
            ("C#", "cs.txt"),
            ("IT", "it.txt"),
            ("OR", "or.sql"),
            ("parse_json", "func.py"),
        ]:
            hits = index.search(query, mode="lexical")
            assert [h.document for h in hits] == [expected]
        assert not index.search("___ + #")


def test_extractive_heading_context_is_opt_in_and_not_citation_text(tmp_path):
    root = tmp_path / "corpus"
    text = (
        "# Treasury\n## Exposure limits\n"
        + "Background material. " * 150
        + "\nMaximum allocation is 7 percent."
    )
    source(root, "manual.md", text)
    with Index(tmp_path / "index.sqlite") as index:
        index.ingest(root, size=300, overlap=30)
        assert all(not h.context for h in index.search("allocation"))
        before = index.status()["generation"]
        assert (
            index.ingest(root, size=300, overlap=30, contextual=True)["generation"]
            == before + 1
        )
        hit = index.search("allocation", k=1)[0]
        assert hit.context == "Treasury > Exposure limits"
        assert hit.text == text[hit.start : hit.end]
        assert index.ingest(root, size=300, overlap=30, contextual=True)["changed"] == 0


def test_v03_migration_preserves_vectors_sources_and_rolls_back_bad_schema(tmp_path):
    from rag_pipeline.index import SCHEMA

    old = SCHEMA.replace(", context TEXT NOT NULL DEFAULT ''", "").replace(
        "tokenize=\"porter unicode61 tokenchars '+#_'\"", "tokenize='porter unicode61'"
    )
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(old)
        db.executemany(
            "INSERT INTO state VALUES(?,?)",
            [
                ("schema", "2"),
                ("generation", "7"),
                ("embedding", Encoder.signature),
                ("dimension", "3"),
            ],
        )
        db.execute("INSERT INTO documents VALUES('code.txt','fingerprint','{}')")
        db.execute("INSERT INTO sections VALUES(1,'code.txt','text','C++ revenue',11)")
        vector = (np.array([1, 0, 1], dtype="<f4") / np.sqrt(2)).astype("<f4")
        db.execute(
            "INSERT INTO chunks VALUES('chunk',1,0,11,'C++ revenue',?)",
            (vector.tobytes(),),
        )
    with Index(path, Encoder()) as index:
        assert index._state("schema") == "3"
        assert index.status()["generation"] == 7
        assert index.search("C++", mode="lexical")[0].text == "C++ revenue"
        assert index.search("revenue", mode="dense")[0].id == "chunk"
        assert (
            index.db.execute("SELECT vector FROM chunks").fetchone()[0]
            == vector.tobytes()
        )
    broken = tmp_path / "partial.sqlite"
    with sqlite3.connect(broken) as db:
        db.executescript(old)
        db.execute("INSERT INTO state VALUES('schema','2')")
        db.execute("DROP TRIGGER chunk_insert")
    with pytest.raises(ValueError, match="Incomplete"):
        Index(broken)
    with sqlite3.connect(broken) as db:
        assert (
            db.execute("SELECT value FROM state WHERE key='schema'").fetchone()[0]
            == "2"
        )
        assert "context" not in {r[1] for r in db.execute("PRAGMA table_info(chunks)")}


def test_dense_block_selection_keeps_boundary_ties_and_scope(tmp_path):
    root = tmp_path / "corpus"
    for name in ("a", "b"):
        source(
            root,
            f"{name}.csv",
            "entry,description\n" + "\n".join(f"{i},Revenue" for i in range(600)),
            scope=name,
        )
    with Index(tmp_path / "index.sqlite", Encoder()) as index:
        index.ingest(root)
        hits = index.search("Revenue", mode="dense", filters={"scope": "b"}, k=5)
        ids = [
            r[0]
            for r in index.db.execute(
                "SELECT c.id FROM chunks c JOIN sections s ON s.id=c.section WHERE s.document='b.csv' ORDER BY c.id LIMIT 5"
            )
        ]
        assert [h.id for h in hits] == ids


def test_query_encoder_is_distinct_from_passage_encoder(tmp_path):
    class Asymmetric(Encoder):
        def encode_queries(self, texts):
            self.questions = texts
            return super().encode(texts)

    model = Asymmetric()
    root = tmp_path / "corpus"
    source(root, "book.txt", "Revenue increased.")
    with Index(tmp_path / "index.sqlite", model) as index:
        index.ingest(root)
        assert not hasattr(model, "questions")
        assert index.search("Revenue", mode="dense")
        assert model.questions == ["Revenue"]


def test_embedding_batches_cross_documents_with_atomic_failure(tmp_path):
    class Batched(Encoder):
        def __init__(self):
            self.batches = []

        def encode(self, texts):
            self.batches.append(len(texts))
            return super().encode(texts)

    root = tmp_path / "corpus"
    for i in range(40):
        source(root, f"{i:02}.txt", "Revenue increased.")
    model = Batched()
    with Index(tmp_path / "index.sqlite", model) as index:
        index.ingest(root)
        assert model.batches == [32, 8]
        before = index.status()
        for p in root.glob("*.txt"):
            p.write_text("Revenue changed.")
        model.encode = lambda texts: np.full((len(texts), 3), np.nan)
        with pytest.raises(ValueError, match="non-finite"):
            index.ingest(root)
        assert index.status() == before
        assert index.search("increased", mode="lexical")


def test_migration_ddl_rolls_back_when_fts_rebuild_fails(tmp_path):
    from rag_pipeline.index import SCHEMA

    old = SCHEMA.replace(", context TEXT NOT NULL DEFAULT ''", "").replace(
        "tokenize=\"porter unicode61 tokenchars '+#_'\"", "tokenize='porter unicode61'"
    )
    path = tmp_path / "bad-content.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(old)
        db.execute("INSERT INTO state VALUES('schema','2')")
        db.execute("ALTER TABLE chunks RENAME COLUMN search_text TO broken_text")
        before = db.execute(
            "SELECT name,type,sql FROM sqlite_master ORDER BY name"
        ).fetchall()
    with pytest.raises(sqlite3.OperationalError):
        Index(path)
    with sqlite3.connect(path) as db:
        assert (
            db.execute("SELECT value FROM state WHERE key='schema'").fetchone()[0]
            == "2"
        )
        assert (
            db.execute(
                "SELECT name,type,sql FROM sqlite_master ORDER BY name"
            ).fetchall()
            == before
        )


def test_dense_cache_matches_streaming_and_invalidates_with_wal_snapshot(tmp_path):
    root = tmp_path / "corpus"
    path = tmp_path / "index.sqlite"
    book = source(root, "book.txt", "Revenue increased.")
    with Index(path, Encoder()) as reader, Index(path, Encoder()) as writer:
        reader.ingest(root)
        old = reader.search("Revenue", mode="dense", min_cosine=0.9)
        assert old and reader._dense_cache[1] is not None
        with Index(path, Encoder(), vector_cache_bytes=0) as streamed:
            assert [
                h.id for h in streamed.search("Revenue", mode="dense", min_cosine=0.9)
            ] == [h.id for h in old]
            assert streamed._dense_cache is None
        original = reader._state
        committed = False

        def concurrent_update(key):
            nonlocal committed
            value = original(key)
            if key == "generation" and not committed:
                committed = True
                book.write_text("Lease obligations.")
                writer.ingest(root)
            return value

        reader._state = concurrent_update
        assert (
            reader.search("Revenue", mode="dense", min_cosine=0.9)[0].text
            == "Revenue increased."
        )
        assert not reader.search("Revenue", mode="dense", min_cosine=0.9)
        assert (
            reader.search("Lease", mode="dense", min_cosine=0.9)[0].text
            == "Lease obligations."
        )
        reader.vector_cache_bytes = 1
        assert reader.search("Lease", mode="dense")
        assert reader._dense_cache[1] is None
        reader.vector_cache_bytes = 0
        assert reader.search("Lease", mode="dense") and reader._dense_cache is None
        reader.vector_cache_bytes = 64 * 1024 * 1024
        assert reader.search("Lease", mode="dense")
        # A direct external vector write changes data_version even without changing
        # the generation marker; a direct local write changes total_changes.
        writer.db.execute(
            "UPDATE chunks SET vector=?", (np.full(3, np.nan, dtype="<f4").tobytes(),)
        )
        with pytest.raises(ValueError, match="Corrupt"):
            reader.search("Lease", mode="dense")
        writer.ingest(root, size=300, overlap=30)
        assert reader.search("Lease", mode="dense")
        reader.db.execute(
            "UPDATE chunks SET vector=?", (np.full(3, np.nan, dtype="<f4").tobytes(),)
        )
        with pytest.raises(ValueError, match="Corrupt"):
            reader.search("Lease", mode="dense")
