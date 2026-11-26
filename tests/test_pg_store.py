"""Postgres store against a real server (set RAG_TEST_POSTGRES to a DSN)."""

import os

import pytest

DSN = os.environ.get("RAG_TEST_POSTGRES")
pytestmark = pytest.mark.skipif(not DSN, reason="Set RAG_TEST_POSTGRES to a DSN")


class Encoder:
    signature = "test:deterministic:v1"

    def encode(self, texts):
        return [
            [
                1.0 if "revenue" in t.lower() else 0.0,
                1.0 if "lease" in t.lower() else 0.0,
                1.0,
            ]
            for t in texts
        ]


@pytest.fixture
def store():
    import psycopg

    from rag_pipeline.pg_store import TABLES, install

    with psycopg.connect(DSN, autocommit=True) as db:
        for table in reversed(TABLES):
            db.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    install(DSN)
    yield DSN


def corpus(root, files):
    root.mkdir(exist_ok=True)
    for name, (text, meta) in files.items():
        (root / name).write_text(text)
        if meta:
            (root / (name + ".meta.json")).write_text(__import__("json").dumps(meta))
    return root


def test_ingest_search_and_incremental_publication(store, tmp_path):
    from rag_pipeline.pg_store import PgIndex

    root = corpus(
        tmp_path / "acme",
        {
            "q1.txt": (
                "Revenue was USD 125.0 million in 2024.",
                {"published": "2024-04-20"},
            ),
            "lease.txt": ("The lease term is ten years.", {"published": "2024-08-01"}),
        },
    )
    with PgIndex(store, "acme", Encoder()) as index:
        stats = index.ingest(root)
        assert stats["documents"] == 2 and stats["generation"] == 1
        hits = index.search(
            "revenue", mode="hybrid", fusion="relative", dense_weight=0.6
        )
        assert hits[0].document == "q1.txt"
        assert hits[0].text == "Revenue was USD 125.0 million in 2024."
        assert [h.document for h in index.search("lease", as_of="2024-07-01")] == []
        assert index.search("revenue", filters={"published": "2024-04-20"})
        # Unchanged sources publish nothing new; a removed file is removed.
        assert index.ingest(root)["changed"] == 0
        (root / "lease.txt").unlink()
        (root / "lease.txt.meta.json").unlink()
        after = index.ingest(root)
        assert after["removed"] == 1 and after["documents"] == 1


def test_row_level_security_isolates_tenants(store, tmp_path):
    import psycopg

    from rag_pipeline.pg_store import PgIndex

    acme = corpus(tmp_path / "acme", {"a.txt": ("Acme revenue is secret.", None)})
    globex = corpus(tmp_path / "globex", {"g.txt": ("Globex lease terms.", None)})
    with PgIndex(store, "acme") as a, PgIndex(store, "globex") as g:
        a.ingest(acme)
        g.ingest(globex)
        assert [h.document for h in a.search("revenue lease")] == ["a.txt"]
        assert [h.document for h in g.search("revenue lease")] == ["g.txt"]
    # Even a raw query on a connection scoped to globex can't read acme rows,
    # and one with no tenant set reads nothing: the database enforces it.
    with psycopg.connect(DSN) as raw:
        raw.execute("SELECT set_config('rag.tenant', 'globex', false)")
        assert raw.execute("SELECT count(*) FROM rag_documents").fetchone()[0] == 1
        raw.execute("SELECT set_config('rag.tenant', '', false)")
        assert raw.execute("SELECT count(*) FROM rag_chunks").fetchone()[0] == 0
    with pytest.raises(ValueError, match="slug"):
        PgIndex(store, "Acme; DROP")


def test_server_engine_on_postgres(store, tmp_path):
    from rag_pipeline.server import Engine, parse_keys

    keys = parse_keys("acme:acme-key-0123456789:admin,globex:globex-key-01234567")
    engine = Engine(tmp_path / "audit", keys, postgres=store)
    docs = corpus(tmp_path / "docs", {"a.txt": ("Acme revenue grew.", None)})
    assert (
        engine.handle("acme", True, "POST", "/v1/ingest", {"directory": str(docs)})[0]
        == 200
    )
    status, found = engine.handle(
        "acme", False, "POST", "/v1/search", {"question": "revenue"}
    )
    assert status == 200 and found["hits"][0]["document"] == "a.txt"
    assert engine.handle(
        "globex", False, "POST", "/v1/search", {"question": "revenue"}
    )[1] == {"hits": []}
