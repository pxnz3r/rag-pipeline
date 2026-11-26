"""Postgres store for many users: row-level security per tenant, pgvector HNSW.

    pip install 'rag-pipeline[postgres]'   # psycopg; the server needs pgvector
    with PgIndex("postgresql://rag@db/rag", tenant="acme", embedder=enc) as index:
        index.ingest("filings/")
        hits = index.search("FY2024 revenue", mode="hybrid", fusion="relative")

Same contract as ``Index``: parsing, chunking, enrichment and encoding stage
through ``index.stage_documents`` (no lock held), then one Postgres
transaction publishes the change under a per-tenant advisory lock, so an
ingest is all-or-nothing. Every table carries ``tenant`` and enforces it with
FORCE ROW LEVEL SECURITY against ``rag.tenant``, set per connection: tenant
scoping lives in the database, not in query code. Connect as a role without
superuser or BYPASSRLS, which skip row security.

Lexical ranking is BM25 over a postings table of stemmed English lexemes
(SciFact NDCG@10 0.687, FTS5 0.682); dense search uses an HNSW index per
vector dimension. Fusion, reranking and evidence windows are the shared code
from ``index``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .index import (
    check_search,
    fuse,
    pipeline_signature,
    query_terms,
    source_paths,
    stage_documents,
    windows,
)

TABLES = (
    "rag_state",
    "rag_documents",
    "rag_document_metadata",
    "rag_sections",
    "rag_chunks",
    "rag_terms",
)
# BM25 parameters, the same as SQLite FTS5's bm25().
K1, B = 1.2, 0.75
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_state(
    tenant text NOT NULL, key text NOT NULL, value text NOT NULL,
    PRIMARY KEY (tenant, key));
CREATE TABLE IF NOT EXISTS rag_documents(
    tenant text NOT NULL, id text NOT NULL, fingerprint text NOT NULL,
    metadata jsonb NOT NULL, PRIMARY KEY (tenant, id));
CREATE TABLE IF NOT EXISTS rag_document_metadata(
    tenant text NOT NULL, document text NOT NULL, key text NOT NULL, value text NOT NULL,
    PRIMARY KEY (tenant, document, key),
    FOREIGN KEY (tenant, document) REFERENCES rag_documents(tenant, id) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS rag_metadata_lookup
    ON rag_document_metadata(tenant, key, value, document);
CREATE TABLE IF NOT EXISTS rag_sections(
    id bigserial PRIMARY KEY, tenant text NOT NULL, document text NOT NULL,
    locator text NOT NULL, text text NOT NULL, chars integer NOT NULL,
    FOREIGN KEY (tenant, document) REFERENCES rag_documents(tenant, id) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS rag_section_document ON rag_sections(tenant, document);
CREATE TABLE IF NOT EXISTS rag_chunks(
    tenant text NOT NULL, id text NOT NULL,
    section bigint NOT NULL REFERENCES rag_sections(id) ON DELETE CASCADE,
    start integer NOT NULL, "end" integer NOT NULL,
    search_text text NOT NULL, context text NOT NULL DEFAULT '',
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    length integer NOT NULL DEFAULT 0,
    vector vector, PRIMARY KEY (tenant, id));
CREATE INDEX IF NOT EXISTS rag_chunks_section ON rag_chunks(section);
-- BM25 postings: stemmed English lexemes from tsv with their counts.
CREATE TABLE IF NOT EXISTS rag_terms(
    tenant text NOT NULL, term text NOT NULL, chunk text NOT NULL, tf integer NOT NULL,
    PRIMARY KEY (tenant, term, chunk),
    FOREIGN KEY (tenant, chunk) REFERENCES rag_chunks(tenant, id) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS rag_terms_chunk ON rag_terms(tenant, chunk);
"""
TENANT = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")


def install(dsn):
    """Create tables, indexes and row-level security (run once, as the owner)."""
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as db:
        db.execute(SCHEMA)
        for table in TABLES:
            db.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            db.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
            db.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
            db.execute(
                f"CREATE POLICY tenant_isolation ON {table} "
                "USING (tenant = current_setting('rag.tenant', true)) "
                "WITH CHECK (tenant = current_setting('rag.tenant', true))"
            )


class PgIndex:
    def __init__(self, dsn, tenant, embedder=None, reranker=None):
        import psycopg
        from psycopg.rows import dict_row

        if not isinstance(tenant, str) or not TENANT.fullmatch(tenant):
            raise ValueError("Tenant must be a lowercase slug")
        signature = getattr(embedder, "signature", "")
        if embedder is not None and (not isinstance(signature, str) or not signature):
            raise ValueError("Embedder requires a stable model/revision signature")
        self.tenant, self.embedder, self.reranker = tenant, embedder, reranker
        self.db = psycopg.connect(dsn, row_factory=dict_row)
        # Session-level setting: every statement on this connection is scoped.
        self.db.execute("SELECT set_config('rag.tenant', %s, false)", (tenant,))
        self.db.commit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.db.close()

    def _state(self, key):
        row = self.db.execute(
            "SELECT value FROM rag_state WHERE key=%s", (key,)
        ).fetchone()
        return row["value"] if row else None

    def status(self):
        with self.db.transaction():
            counts = {
                name: self.db.execute(
                    f"SELECT count(*) AS n FROM rag_{name}"
                ).fetchone()["n"]
                for name in ("documents", "sections", "chunks")
            }
            return {
                **counts,
                "generation": int(self._state("generation") or 0),
                "embedding": self._state("embedding") or None,
                "plan": json.loads(self._state("plan") or "null"),
            }

    def ingest(
        self,
        root,
        *,
        size=900,
        overlap=100,
        contextual=False,
        plan=None,
        parser="pdfium",
        enricher=None,
    ):
        root = Path(root).resolve()
        paths = source_paths(root, size, overlap, contextual, parser)
        encoder = getattr(self.embedder, "signature", "")
        pipeline = pipeline_signature(
            size, overlap, encoder, contextual, parser, enricher
        )
        with self.db.transaction():
            generation = self._state("generation")
            if self._state("embedding") and not self.embedder:
                raise ValueError(
                    "This index has vectors; supply the same embedder to ingest"
                )
            rebuild = self._state("pipeline") != pipeline
            existing = {
                r["id"]: r["fingerprint"]
                for r in self.db.execute("SELECT id, fingerprint FROM rag_documents")
            }
            dimension = (
                self._state("dimension")
                if self._state("embedding") == encoder
                else None
            )
        cache = (
            Path.home() / ".cache" / "rag-pipeline" / f"pg-{self.tenant}.enrich-cache"
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        with stage_documents(
            paths,
            root,
            self.embedder,
            existing=existing,
            rebuild=rebuild,
            dimension=dimension,
            size=size,
            overlap=overlap,
            contextual=contextual,
            parser=parser,
            enricher=enricher,
            cache_path=cache,
        ) as (staged_path, seen, changed, staged_dimension):
            removed = existing.keys() - seen
            self._publish(
                staged_path,
                removed,
                changed,
                generation,
                pipeline,
                encoder,
                staged_dimension,
                plan,
            )
        return {"changed": changed, "removed": len(removed), **self.status()}

    def _publish(
        self,
        staged_path,
        removed,
        changed,
        generation,
        pipeline,
        encoder,
        dimension,
        plan,
    ):
        import sqlite3

        stage = sqlite3.connect(staged_path)
        try:
            with self.db.transaction():
                # One writer per tenant; a peer that published meanwhile wins.
                self.db.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))", (self.tenant,)
                )
                if self._state("generation") != generation:
                    raise ValueError(
                        "Index changed during ingestion preparation; retry"
                    )
                staged_ids = [r[0] for r in stage.execute("SELECT id FROM documents")]
                gone = list(removed) + staged_ids
                if gone:
                    self.db.execute(
                        "DELETE FROM rag_documents WHERE id = ANY(%s)", (gone,)
                    )
                with self.db.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO rag_documents VALUES (%s, %s, %s, %s::jsonb)",
                        [
                            (self.tenant, i, f, m)
                            for i, f, m in stage.execute(
                                "SELECT id, fingerprint, metadata FROM documents"
                            )
                        ],
                    )
                    cursor.executemany(
                        "INSERT INTO rag_document_metadata VALUES (%s, %s, %s, %s)",
                        [
                            (self.tenant, *row)
                            for row in stage.execute(
                                "SELECT document, key, value FROM document_metadata"
                            )
                        ],
                    )
                    sections = {}
                    for old, document, locator, text, chars in stage.execute(
                        "SELECT id, document, locator, text, chars FROM sections"
                    ):
                        sections[old] = self.db.execute(
                            "INSERT INTO rag_sections(tenant, document, locator, text, chars) "
                            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                            (self.tenant, document, locator, text, chars),
                        ).fetchone()["id"]
                    cursor.executemany(
                        'INSERT INTO rag_chunks(tenant, id, section, start, "end", '
                        "search_text, context, vector) VALUES (%s, %s, %s, %s, %s, %s, %s, "
                        "%s::vector)",
                        [
                            (
                                self.tenant,
                                cid,
                                sections[section],
                                start,
                                end,
                                text,
                                context,
                                _vector_literal(vector),
                            )
                            for cid, section, start, end, text, vector, context in stage.execute(
                                "SELECT id, section, start, end, search_text, vector, context "
                                "FROM chunks"
                            )
                        ],
                    )
                chunk_ids = [r[0] for r in stage.execute("SELECT id FROM chunks")]
                if chunk_ids:
                    # Postings and lengths for BM25, from the stored tsvector.
                    self.db.execute(
                        "INSERT INTO rag_terms(tenant, term, chunk, tf) "
                        "SELECT c.tenant, t.lexeme, c.id, "
                        "coalesce(array_length(t.positions, 1), 1) "
                        "FROM rag_chunks c, unnest(c.tsv) t WHERE c.id = ANY(%s)",
                        (chunk_ids,),
                    )
                    self.db.execute(
                        "UPDATE rag_chunks c SET length = coalesce((SELECT sum(tf) "
                        "FROM rag_terms t WHERE t.chunk = c.id), 0) WHERE c.id = ANY(%s)",
                        (chunk_ids,),
                    )
                totals = self.db.execute(
                    "SELECT count(*) AS n, coalesce(avg(length), 0) AS mean FROM rag_chunks"
                ).fetchone()
                state = {
                    "pipeline": pipeline,
                    "embedding": encoder,
                    "chunk_count": str(totals["n"]),
                    "mean_length": str(float(totals["mean"])),
                }
                if dimension:
                    state["dimension"] = dimension
                if plan:
                    state["plan"] = json.dumps(plan, sort_keys=True)
                if changed or removed:
                    state["generation"] = str(int(generation or 0) + 1)
                for key, value in state.items():
                    self.db.execute(
                        "INSERT INTO rag_state VALUES (%s, %s, %s) "
                        "ON CONFLICT (tenant, key) DO UPDATE SET value = EXCLUDED.value",
                        (self.tenant, key, value),
                    )
                if dimension:
                    # HNSW needs a fixed dimension; one index per dimension in use.
                    d = int(dimension)
                    self.db.execute(
                        f"CREATE INDEX IF NOT EXISTS rag_chunks_hnsw_{d} ON rag_chunks "
                        f"USING hnsw ((vector::vector({d})) vector_cosine_ops) "
                        f"WHERE vector_dims(vector) = {d}"
                    )
        finally:
            stage.close()

    def _filters(self, filters):
        from .sources import metadata

        parts, args = [], []
        for key, value in metadata({} if filters is None else filters).items():
            parts.append(
                "s.document IN (SELECT document FROM rag_document_metadata "
                "WHERE key=%s AND value=%s)"
            )
            args.extend([key, str(value)])
        return "".join(" AND " + p for p in parts), args

    def _bm25(self, text, match, clause, args, candidates):
        """(chunk id, BM25) best first, over stemmed English lexemes.

        Postgres's ts_rank_cd scored SciFact NDCG@10 0.430 against FTS5 BM25's
        0.682, so lexical ranking computes BM25 from the postings table:
        idf = ln(1 + (N - n + 0.5) / (n + 0.5)), k1 = 1.2, b = 0.75.
        """
        count = float(self._state("chunk_count") or 0)
        mean = float(self._state("mean_length") or 0) or 1.0
        having = " HAVING count(*) = (SELECT count(*) FROM q)" if match == "all" else ""
        rows = self.db.execute(
            "WITH q AS (SELECT DISTINCT unnest(tsvector_to_array("
            "to_tsvector('english', %s))) AS term), "
            "df AS (SELECT t.term, count(*) AS n FROM rag_terms t JOIN q USING (term) "
            "GROUP BY t.term) "
            "SELECT t.chunk AS id, sum(ln(1 + (%s - df.n + 0.5) / (df.n + 0.5)) "
            "* t.tf * %s / (t.tf + %s * (1 - %s + %s * c.length / %s))) AS score "
            "FROM rag_terms t JOIN df USING (term) JOIN rag_chunks c ON c.id = t.chunk "
            "JOIN rag_sections s ON s.id = c.section WHERE true"
            + clause
            + " GROUP BY t.chunk"
            + having
            + " ORDER BY score DESC, t.chunk LIMIT %s",
            [text, count, K1 + 1, K1, B, B, mean, *args, candidates],
        )
        return [(r["id"], float(r["score"])) for r in rows]

    def _text(self, section, start, end):
        return self.db.execute(
            "SELECT substr(text, %s, %s) AS t FROM rag_sections WHERE id=%s",
            (start + 1, end - start, section),
        ).fetchone()["t"]

    def search(
        self,
        question,
        *,
        filters=None,
        k=5,
        candidates=40,
        mode="lexical",
        min_cosine=0.3,
        match="any",
        context_chars=1800,
        diversify=True,
        distinct_documents=False,
        fusion="rrf",
        dense_weight=0.5,
        rerank_candidates=None,
        as_of=None,
    ):
        import numpy as np

        rerank_candidates = (
            candidates if rerank_candidates is None else rerank_candidates
        )
        check_search(
            question,
            k=k,
            candidates=candidates,
            mode=mode,
            min_cosine=min_cosine,
            match=match,
            context_chars=context_chars,
            diversify=diversify,
            distinct_documents=distinct_documents,
            fusion=fusion,
            dense_weight=dense_weight,
            rerank_candidates=rerank_candidates,
            as_of=as_of,
        )
        if mode != "lexical" and not self.embedder:
            raise ValueError("Dense/hybrid search requires an embedder")
        terms = query_terms(question)
        if not terms:
            return []
        clause, args = self._filters(filters)
        if as_of is not None:
            clause += (
                " AND s.document IN (SELECT document FROM rag_document_metadata "
                "WHERE key='published' AND (CASE WHEN length(value)=10 "
                "THEN value||'T23:59:59' ELSE value END) <= %s)"
            )
            args.append(as_of if "T" in as_of else as_of + "T23:59:59")
        query = None
        if mode != "lexical":
            vectors = np.asarray(
                self.embedder.encode_queries([question])
                if hasattr(self.embedder, "encode_queries")
                else self.embedder.encode([question]),
                dtype="<f4",
            )
            query = vectors[0] / np.linalg.norm(vectors[0])
        with self.db.transaction():
            # A consistent snapshot for both channels and every window read.
            self.db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            if (
                mode != "lexical"
                and self._state("embedding") != self.embedder.signature
            ):
                raise ValueError(
                    "Query embedding model does not match the indexed model"
                )
            lexical, dense = [], []
            if mode != "dense":
                lexical = self._bm25(" ".join(terms), match, clause, args, candidates)
            if query is not None:
                d = len(query)
                if self._state("dimension") != str(d):
                    raise ValueError("Embedding dimensions changed; create a new index")
                literal = _vector_literal(query.astype("<f4").tobytes())
                dense = [
                    (r["id"], r["score"])
                    for r in self.db.execute(
                        f"SELECT c.id, 1 - ((c.vector::vector({d})) <=> %s::vector({d})) "
                        "AS score FROM rag_chunks c JOIN rag_sections s ON s.id = c.section "
                        f"WHERE vector_dims(c.vector) = {d}"
                        + clause
                        + f" ORDER BY (c.vector::vector({d})) <=> %s::vector({d}), c.id "
                        "LIMIT %s",
                        [literal, *args, literal, candidates],
                    )
                    if r["score"] >= min_cosine
                ]
            order, scores = fuse(lexical, dense, fusion, dense_weight, candidates)
            if not order:
                return []
            rows = {
                r["id"]: r
                for r in self.db.execute(
                    'SELECT c.id, c.section, c.start, c."end" AS end, c.context, '
                    "s.document, s.locator, s.chars, d.metadata, d.fingerprint "
                    "FROM rag_chunks c JOIN rag_sections s ON s.id = c.section "
                    "JOIN rag_documents d ON d.id = s.document WHERE c.id = ANY(%s)",
                    (order,),
                )
            }
            if self.reranker:
                head, tail = order[:rerank_candidates], order[rerank_candidates:]
                passages = [
                    (rows[c]["context"] + "\n" if rows[c]["context"] else "")
                    + self._text(rows[c]["section"], rows[c]["start"], rows[c]["end"])
                    for c in head
                ]
                ranked = np.asarray(
                    self.reranker.score(question, passages), dtype=float
                )
                if ranked.shape != (len(head),) or not np.isfinite(ranked).all():
                    raise ValueError("Invalid reranker scores")
                scores = dict(zip(head, ranked.tolist()))
                head.sort(key=lambda c: (-scores[c], c))
                floor = min(scores.values()) - 1
                scores.update({c: floor for c in tail})
                order = head + tail
            return windows(
                order,
                rows,
                scores,
                self._text,
                k=k,
                context_chars=context_chars,
                diversify=diversify,
                distinct_documents=distinct_documents,
            )


def _vector_literal(blob):
    """pgvector text form from a little-endian float32 blob, or None."""
    if blob is None:
        return None
    import numpy as np

    values = np.frombuffer(blob, dtype="<f4")
    return "[" + ",".join(repr(float(v)) for v in values) + "]"
