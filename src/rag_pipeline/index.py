"""One transactional source of truth for text, filters, lexical and dense retrieval."""

from __future__ import annotations

import heapq
import json
import math
import os
import re
import sqlite3
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .sources import (
    FORMATS,
    HEADING_FORMATS,
    PARSERS,
    Document,
    contextual_spans,
    document,
    headings,
    metadata,
    spans,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, metadata TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS document_metadata(document TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(document,key));
CREATE INDEX IF NOT EXISTS metadata_lookup ON document_metadata(key,value,document);
CREATE TABLE IF NOT EXISTS sections(id INTEGER PRIMARY KEY, document TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, locator TEXT NOT NULL, text TEXT NOT NULL, chars INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(id TEXT UNIQUE NOT NULL, section INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE, start INTEGER NOT NULL, end INTEGER NOT NULL, search_text TEXT NOT NULL, vector BLOB, context TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS section_document ON sections(document);
CREATE INDEX IF NOT EXISTS chunk_section ON chunks(section);
"""
FTS_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(search_text, content=chunks, content_rowid=rowid, tokenize=\"porter unicode61 tokenchars '+#_'\")",
    "CREATE TRIGGER IF NOT EXISTS chunk_insert AFTER INSERT ON chunks BEGIN INSERT INTO fts(rowid,search_text) VALUES(new.rowid,new.search_text); END",
    "CREATE TRIGGER IF NOT EXISTS chunk_delete AFTER DELETE ON chunks BEGIN INSERT INTO fts(fts,rowid,search_text) VALUES('delete',old.rowid,old.search_text); END",
)
SCHEMA += ";\n".join(FTS_DDL) + ";\n"
STOP = set(
    "a an and are as at be by can do does for from how i in is it of on or that the their this to was were what when where which who why with would".split()
)


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?")


def _walk_error(error):
    raise error


@contextmanager
def _closing(resource):
    try:
        yield resource
    finally:
        if resource is not None:
            resource.close()


def _parse(paths, root, existing, parser="pdfium"):
    """Documents in path order, parsed a bounded window ahead on worker threads.

    PDF extraction runs in subprocesses, so threads overlap it with encoding
    and with each other. The window bounds how many parsed documents wait in
    memory; an error cancels the queued work and surfaces in order.
    """
    workers = min(8, os.cpu_count() or 1)
    pool = ThreadPoolExecutor(workers, thread_name_prefix="rag-parse")
    pending = deque()
    try:
        for path in paths:
            previous = (
                existing.get(path.relative_to(root).as_posix()) if existing else None
            )
            pending.append(pool.submit(document, path, root, previous, parser))
            if len(pending) > 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class Hit:
    id: str
    document: str
    locator: str
    start: int
    end: int
    text: str
    metadata: dict
    score: float
    context: str = ""
    source_revision: str = ""
    source_prefix: str = ""
    source_suffix: str = ""

    def to_dict(self):
        return asdict(self)


class Index:
    def __init__(
        self,
        path: str | Path,
        embedder=None,
        reranker=None,
        *,
        # 64 MiB held only ~20k 768-d vectors; past it each dense query
        # streamed from SQLite (332 ms vs 32 ms on 82k FiQA chunks).
        vector_cache_bytes=512 * 1024 * 1024,
        shared_threads=False,
    ):
        if (
            not isinstance(vector_cache_bytes, int)
            or isinstance(vector_cache_bytes, bool)
            or not 0 <= vector_cache_bytes <= 1024**3
        ):
            raise ValueError("Vector cache budget must be between 0 and 1 GiB")
        self.vector_cache_bytes = vector_cache_bytes
        self._dense_cache = None
        signature = getattr(embedder, "signature", "")
        if embedder is not None and (
            not isinstance(signature, str) or not signature.strip()
        ):
            raise ValueError("Embedder requires a stable model/revision signature")
        self.path = Path(path)
        existed = self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # shared_threads lets a caller that serializes access (the server holds
        # a lock per index) use this connection from several threads.
        self.db = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=not shared_threads,
        )
        self.db.row_factory = sqlite3.Row
        self.embedder = embedder
        self.reranker = reranker
        try:
            if not existed and os.name == "posix":
                self.path.chmod(0o600)
            if existed and self._state("schema") not in {"2", "3"}:
                raise ValueError(
                    "Missing or unsupported index schema; create a new index"
                )
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            if existed:
                tables = {
                    r[0]
                    for r in self.db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if (
                    not {
                        "state",
                        "documents",
                        "document_metadata",
                        "sections",
                        "chunks",
                        "fts",
                    }
                    <= tables
                ):
                    raise ValueError("Incomplete index schema; restore a valid backup")
            else:
                self.db.executescript(SCHEMA)
            version = self._state("schema")
            triggers = {
                r[0]
                for r in self.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
            columns = {r[1] for r in self.db.execute("PRAGMA table_info(chunks)")}
            if not {"chunk_insert", "chunk_delete"} <= triggers or (
                version == "3" and "context" not in columns
            ):
                raise ValueError("Incomplete index schema; restore a valid backup")
            if version == "2":
                # Preserve sources, vectors and generation in one atomic migration.
                with self._transaction(write=True):
                    if self._state("schema") == "2":
                        self.db.execute(
                            "ALTER TABLE chunks ADD COLUMN context TEXT NOT NULL DEFAULT ''"
                        )
                        for name in ("chunk_insert", "chunk_delete"):
                            self.db.execute(f"DROP TRIGGER {name}")
                        self.db.execute("DROP TABLE fts")
                        for statement in FTS_DDL:
                            self.db.execute(statement)
                        self.db.execute("INSERT INTO fts(fts) VALUES('rebuild')")
                        self.db.execute("UPDATE state SET value='3' WHERE key='schema'")
            if version not in (None, "2", "3"):
                raise ValueError("Unsupported index schema; create a new index")
            if version is None:
                self.db.execute("INSERT INTO state VALUES('schema','3')")
        except Exception:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._dense_cache = None
        self.db.close()

    def _state(self, key):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    @contextmanager
    def _transaction(self, write=False):
        if self.db.in_transaction:
            if write:
                raise ValueError("Cannot nest an index write transaction")
            yield
            return
        self.db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def status(self):
        with self._transaction():
            return {
                **{
                    name: self.db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                    for name in ("documents", "sections", "chunks")
                },
                "generation": int(self._state("generation") or 0),
                "embedding": self._state("embedding") or None,
                "plan": json.loads(self._state("plan") or "null"),
            }

    def ingest(
        self,
        root: str | Path,
        *,
        size=900,
        overlap=100,
        contextual=False,
        plan: dict | None = None,
        parser: str = "pdfium",
        enricher=None,
    ):
        """Synchronize ``root`` atomically; ``plan`` records the resolved stages.

        ``parser`` picks the PDF extractor and ``enricher`` (an
        ``enrich.ChatEnricher``) writes chunk context; both enter the pipeline
        signature, so switching either rebuilds instead of mixing formats.
        """
        if parser not in PARSERS:
            raise ValueError(f"Unknown PDF parser {parser!r}")
        if not isinstance(contextual, bool):
            raise ValueError("Contextual must be a boolean")
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(
                "Source directory must exist; refusing to prune an absent corpus"
            )
        # Validate before a no-op ingest as well.
        list(spans("", size, overlap))
        paths = []
        for directory, subdirs, names in os.walk(
            root, onerror=_walk_error, followlinks=False
        ):
            if any((Path(directory) / name).is_symlink() for name in subdirs):
                raise ValueError("Symlinked source directories are not supported")
            for name in names:
                path = Path(directory) / name
                if path.suffix.lower() not in FORMATS or name.endswith(".meta.json"):
                    continue
                if path.is_symlink() or not path.is_file():
                    raise ValueError(
                        "Symlinked or unreadable source files are not supported"
                    )
                paths.append(path)
        paths.sort()
        signature = getattr(self.embedder, "signature", "")
        # Heading extraction is versioned separately so plain indexes keep their state.
        pipeline = json.dumps(
            [
                "extract-v5",
                size,
                overlap,
                signature,
                contextual and "headings-v2",
                *([] if parser == "pdfium" else [f"pdf:{parser}"]),
                *([] if enricher is None else [f"enrich:{enricher.signature}"]),
            ]
        )
        if self.db.in_transaction:
            raise ValueError("Ingest cannot run inside another transaction")
        with self._transaction():
            expected = (
                self._state("generation"),
                self._state("pipeline"),
                self._state("embedding"),
                self.db.execute("PRAGMA data_version").fetchone()[0],
                self.db.total_changes,
            )
            if self._state("embedding") and not self.embedder:
                raise ValueError(
                    "This index has vectors; supply the same embedder to ingest"
                )
            rebuild = self._state("pipeline") != pipeline
            existing = dict(self.db.execute("SELECT id,fingerprint FROM documents"))
            dimension = (
                self._state("dimension")
                if self._state("embedding") == signature
                else None
            )
        # Parse and infer into a bounded on-disk stage, with no live writer lock.
        # Only changed documents are staged; unchanged indexes are not cloned.
        cache = None
        if enricher is not None:
            from .enrich import Cache

            cache = Cache(self.path.with_name(self.path.name + ".enrich-cache"))
        with (
            tempfile.TemporaryDirectory(prefix="rag-ingest-") as directory,
            _closing(cache),
        ):
            staged_path = Path(directory) / "stage.sqlite"
            with Index(staged_path, self.embedder, vector_cache_bytes=0) as staged:
                seen, changed, batch = set(), 0, []
                with staged._transaction(write=True):
                    if contextual:
                        staged._heading_schema()
                    if dimension:
                        staged.db.execute(
                            "INSERT OR REPLACE INTO state VALUES('dimension',?)",
                            (dimension,),
                        )
                    for doc in _parse(
                        paths, root, None if rebuild else existing, parser
                    ):
                        seen.add(doc.id)
                        if not rebuild and existing.get(doc.id) == doc.fingerprint:
                            continue
                        staged._replace(
                            doc, size, overlap, contextual, batch, enricher, cache
                        )
                        changed += 1
                    staged._insert_chunks(batch)
                staged_dimension = staged._state("dimension")
            removed = existing.keys() - seen
            self.db.execute("ATTACH DATABASE ? AS ingest_stage", (str(staged_path),))
            try:
                with self._transaction(write=True):
                    current = (
                        self._state("generation"),
                        self._state("pipeline"),
                        self._state("embedding"),
                        self.db.execute("PRAGMA data_version").fetchone()[0],
                        self.db.total_changes,
                    )
                    if current != expected:
                        raise ValueError(
                            "Index changed during ingestion preparation; retry"
                        )
                    if contextual:
                        self._heading_schema()
                    offset = self.db.execute(
                        "SELECT coalesce(max(id),0) FROM sections"
                    ).fetchone()[0]
                    self.db.execute(
                        "DELETE FROM documents WHERE id IN (SELECT id FROM ingest_stage.documents)"
                    )
                    self.db.executemany(
                        "DELETE FROM documents WHERE id=?", [(i,) for i in removed]
                    )
                    self.db.execute(
                        "INSERT INTO documents SELECT * FROM ingest_stage.documents"
                    )
                    self.db.execute(
                        "INSERT INTO document_metadata SELECT * FROM ingest_stage.document_metadata"
                    )
                    self.db.execute(
                        "INSERT INTO sections SELECT id+?,document,locator,text,chars FROM ingest_stage.sections",
                        (offset,),
                    )
                    self.db.execute(
                        "INSERT INTO chunks SELECT id,section+?,start,end,search_text,vector,context FROM ingest_stage.chunks",
                        (offset,),
                    )
                    if contextual:
                        self.db.execute(
                            """INSERT INTO heading_nodes
                            SELECT cast(section+? AS text)||substr(id,instr(id,':')),section+?,
                            CASE WHEN parent IS NULL THEN NULL ELSE cast(section+? AS text)||substr(parent,instr(parent,':')) END,
                            start,end,level,title FROM ingest_stage.heading_nodes ORDER BY section,start""",
                            (offset, offset, offset),
                        )
                    if staged_dimension:
                        self.db.execute(
                            "INSERT OR REPLACE INTO state VALUES('dimension',?)",
                            (staged_dimension,),
                        )
                    elif expected[2] != signature:
                        self.db.execute("DELETE FROM state WHERE key='dimension'")
                    for key, value in [
                        ("pipeline", pipeline),
                        ("embedding", signature),
                    ] + ([("plan", json.dumps(plan, sort_keys=True))] if plan else []):
                        self.db.execute(
                            "INSERT OR REPLACE INTO state VALUES(?,?)", (key, value)
                        )
                    if changed or removed:
                        self.db.execute(
                            "INSERT OR REPLACE INTO state VALUES('generation',?)",
                            (str(int(expected[0] or 0) + 1),),
                        )
                        self._dense_cache = None
            finally:
                self.db.execute("DETACH DATABASE ingest_stage")
        return {"changed": changed, "removed": len(removed), **self.status()}

    def _heading_schema(self):
        self.db.execute("""CREATE TABLE IF NOT EXISTS heading_nodes(
            id TEXT PRIMARY KEY, section INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
            parent TEXT REFERENCES heading_nodes(id) ON DELETE CASCADE,
            start INTEGER NOT NULL,end INTEGER NOT NULL,level INTEGER NOT NULL,title TEXT NOT NULL)""")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS heading_section ON heading_nodes(section,parent,start)"
        )

    def _replace(
        self,
        doc: Document,
        size: int,
        overlap: int,
        contextual: bool,
        batch: list,
        enricher=None,
        cache=None,
    ):
        self.db.execute("DELETE FROM documents WHERE id=?", (doc.id,))
        self.db.execute(
            "INSERT INTO documents VALUES(?,?,?)",
            # Sidecars allow 32 fields; the implicit default title may be the 33rd.
            (
                doc.id,
                doc.fingerprint,
                json.dumps(metadata(doc.metadata, max_fields=33)),
            ),
        )
        self.db.executemany(
            "INSERT INTO document_metadata VALUES(?,?,?)",
            [(doc.id, key, str(value)) for key, value in doc.metadata.items()],
        )
        title = " ".join(str(v) for v in doc.metadata.values())
        contextual = contextual and Path(doc.id).suffix.lower() in HEADING_FORMATS
        # The opening of the document (cover, contents) names the entity and
        # period that every later chunk belongs to.
        opening = "\n".join(text for _, text in doc.sections)[:6000]
        for locator, text in doc.sections:
            section = self.db.execute(
                "INSERT INTO sections(document,locator,text,chars) VALUES(?,?,?,?)",
                (doc.id, locator, text, len(text)),
            ).lastrowid
            if contextual:
                active = []
                for start, level, heading_title, _ in headings(text):
                    while active and active[-1][0] >= level:
                        _, node = active.pop()
                        self.db.execute(
                            "UPDATE heading_nodes SET end=? WHERE id=?", (start, node)
                        )
                    node = f"{section}:{start}"
                    self.db.execute(
                        "INSERT INTO heading_nodes VALUES(?,?,?,?,?,?,?)",
                        (
                            node,
                            section,
                            active[-1][1] if active else None,
                            start,
                            len(text),
                            level,
                            heading_title,
                        ),
                    )
                    active.append((level, node))
            spans_here = list(contextual_spans(text, size, overlap, contextual))
            if enricher is not None and spans_here:
                excerpt = opening + "\n...\n" + text[:4000]
                written = enricher.contexts(
                    excerpt, [text[a:b] for a, b, _ in spans_here], cache
                )
                spans_here = [
                    (a, b, "\n".join(filter(None, (context, extra))))
                    for (a, b, context), extra in zip(spans_here, written)
                ]
            for a, b, context in spans_here:
                batch.append(
                    (
                        f"{doc.id}#{locator}:{a}-{b}",
                        section,
                        a,
                        b,
                        (
                            title[:300] + "\n" + context + "\n"
                            if context
                            else title[:300] + "\n"
                        )
                        + text[a:b],
                        context,
                    )
                )
                # A wide window lets encoders sort by length and pad tightly.
                if len(batch) == 256:
                    self._insert_chunks(batch)
                    batch.clear()

    def _insert_chunks(self, batch):
        if not batch:
            return
        vectors = (
            self._vectors([row[4] for row in batch])
            if self.embedder
            else [None] * len(batch)
        )
        self.db.executemany(
            "INSERT INTO chunks VALUES(?,?,?,?,?,?,?)",
            [
                (*row[:5], v.tobytes() if v is not None else None, row[5])
                for row, v in zip(batch, vectors)
            ],
        )

    def _text(self, section, start, end):
        return self.db.execute(
            "SELECT substr(text,?,?) FROM sections WHERE id=?",
            (start + 1, end - start, section),
        ).fetchone()[0]

    def _passage(self, row):
        return (row["context"] + "\n" if row["context"] else "") + self._text(
            row["section"], row["start"], row["end"]
        )

    def prepare_reranker_cache(self):
        from .token_cache import prepare

        return prepare(self)

    def _vectors(self, texts, write=True):
        import numpy as np

        encode = (
            self.embedder.encode
            if write
            else getattr(self.embedder, "encode_queries", self.embedder.encode)
        )
        values = np.asarray(encode(texts), dtype="<f4")
        if (
            values.ndim != 2
            or values.shape[0] != len(texts)
            or not values.shape[1]
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid embedding shape or non-finite values")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms == 0).any() or not np.isfinite(norms).all():
            raise ValueError("Zero-length embedding")
        dimension = str(values.shape[1])
        if self._state("dimension") not in (None, dimension):
            raise ValueError("Embedding dimensions changed; create a new index")
        if write:
            self.db.execute(
                "INSERT OR IGNORE INTO state VALUES('dimension',?)", (dimension,)
            )
        elif self._state("dimension") != dimension:
            raise ValueError("Missing embedding dimension")
        return values / norms

    def _dense_ranking(self, query, clause, args, candidates, min_cosine):
        """(chunk id, cosine) pairs, best first; ties break on id."""
        import numpy as np

        def top(matrix, ids):
            # BLAS finds the neighbourhood (7x faster than einsum on 82k x 768),
            # but its rounding depends on the matrix shape. Rescoring only that
            # neighbourhood with row-wise einsum makes every cosine bitwise
            # identical on cached, filtered and streamed paths, so ties never
            # depend on the path. The slack is far above float32 disagreement
            # (under 1e-7 measured for unit vectors), so no true winner is lost.
            slack = 1e-3
            rough = matrix @ query
            keep = np.flatnonzero(rough >= min_cosine - slack)
            if len(keep) > candidates:
                cutoff = np.partition(rough[keep], -candidates)[-candidates]
                keep = keep[rough[keep] >= cutoff - 2 * slack]
            scores = np.einsum("ij,j->i", matrix[keep], query)
            positions = np.flatnonzero(scores >= min_cosine)
            if len(positions) > candidates:
                cutoff = np.partition(scores[positions], -candidates)[-candidates]
                positions = positions[scores[positions] >= cutoff]
            return heapq.nsmallest(
                candidates,
                [(-float(scores[i]), ids[keep[i]]) for i in positions],
            )

        def unpack(blobs):
            # One buffer per block: per-row frombuffer/stack dominated scan time.
            block = np.frombuffer(b"".join(blobs), dtype="<f4")
            if block.size != len(blobs) * len(query):
                raise ValueError("Corrupt stored vectors")
            block = block.reshape(len(blobs), len(query))
            if not np.isfinite(block).all() or not np.allclose(
                np.einsum("ij,ij->i", block, block), 1, atol=2e-3
            ):
                raise ValueError("Corrupt stored vectors")
            return block

        scope = (
            "FROM chunks c JOIN sections s ON s.id=c.section "
            "JOIN documents d ON d.id=s.document WHERE c.vector IS NOT NULL" + clause
        )
        cached = self._dense_matrix(len(query), unpack)
        if cached is not None:
            matrix, ids, rowids = cached
            if not clause:
                return [(c, -s) for s, c in top(matrix, ids)]
            # Scope selects rows of the cached matrix instead of streaming blobs.
            allowed = np.fromiter(
                (r[0] for r in self.db.execute("SELECT c.rowid " + scope, args)),
                dtype=np.int64,
            )
            positions = np.searchsorted(rowids, allowed)
            if len(allowed) and (
                positions.max() >= len(rowids) or (rowids[positions] != allowed).any()
            ):
                raise ValueError("Dense cache does not match the index snapshot")
            best = top(matrix[positions], [ids[p] for p in positions])
            return [(c, -s) for s, c in best]
        cursor = self.db.execute("SELECT c.id,c.vector " + scope, args)
        best = []
        while rows := cursor.fetchmany(4096):
            block = unpack([r[1] for r in rows])
            best = heapq.nsmallest(candidates, best + top(block, [r[0] for r in rows]))
        return [(c, -s) for s, c in best]

    def _dense_matrix(self, dimension, unpack):
        """All stored vectors in memory when they fit the budget, else None.

        Keyed on the generation, this connection's writes and SQLite's data
        version, so any committed change rebuilds it on the next query.
        """
        import numpy as np

        if not self.vector_cache_bytes:
            self._dense_cache = None
            return None
        token = (
            self._state("generation"),
            self.db.total_changes,
            self.db.execute("PRAGMA data_version").fetchone()[0],
            dimension,
            self.vector_cache_bytes,
        )
        if self._dense_cache is not None and self._dense_cache[0] == token:
            return self._dense_cache[1]
        self._dense_cache = (token, None)
        count, chars = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(length(id)),0) FROM chunks"
        ).fetchone()
        # Conservative allowance for IDs, rowids, references, scores and selection
        # temporaries; model/native/SQLite allocations are separate.
        if count * (4 * dimension + 264) + 4 * chars > self.vector_cache_bytes:
            return None
        matrix = np.empty((count, dimension), dtype="<f4")
        rowids = np.empty(count, dtype=np.int64)
        ids, offset = [], 0
        cursor = self.db.execute(
            "SELECT rowid,id,vector FROM chunks WHERE vector IS NOT NULL ORDER BY rowid"
        )
        while rows := cursor.fetchmany(4096):
            matrix[offset : offset + len(rows)] = unpack([r[2] for r in rows])
            rowids[offset : offset + len(rows)] = [r[0] for r in rows]
            ids.extend(r[1] for r in rows)
            offset += len(rows)
        self._dense_cache = (token, (matrix[:offset], ids, rowids[:offset]))
        return self._dense_cache[1]

    @staticmethod
    def _filters(filters):
        parts, args = [], []
        for key, value in metadata({} if filters is None else filters).items():
            parts.append(
                "d.id IN (SELECT document FROM document_metadata WHERE key=? AND value=?)"
            )
            args.extend([key, str(value)])
        return (" AND " + " AND ".join(parts) if parts else ""), args

    def search(
        self,
        question: str,
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
    ) -> list[Hit]:
        """Rank chunks, then return up to ``k`` original evidence windows.

        Each channel retrieves ``candidates`` chunks. ``fusion="rrf"`` adds
        weighted reciprocal ranks; ``"relative"`` adds min-max normalized
        scores (BM25 and cosine), weighted by ``dense_weight``. A reranker
        rescoring the top ``rerank_candidates`` fused chunks decides their
        order; the rest keep the fused order behind them.
        """
        if rerank_candidates is None:
            rerank_candidates = candidates
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
        if mode in {"dense", "hybrid"} and not self.embedder:
            raise ValueError("Dense/hybrid search requires an embedder")
        terms = query_terms(question)
        if not terms:
            return []
        clause, args = self._filters(filters)
        if as_of is not None:
            # Point-in-time retrieval: only documents whose "published"
            # metadata is on or before as_of. An undated document can't be
            # shown to predate as_of, so it is excluded (no look-ahead).
            # A date-only value counts as the end of that day, so a filing
            # dated the same day as an intraday as_of stays excluded.
            clause += (
                " AND d.id IN (SELECT document FROM document_metadata "
                "WHERE key='published' AND (CASE WHEN length(value)=10 "
                "THEN value||'T23:59:59' ELSE value END)<=?)"
            )
            args = [*args, as_of if "T" in as_of else as_of + "T23:59:59"]
        query = None
        if mode != "lexical":
            if not self.db.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
                return []
            if self._state("embedding") != self.embedder.signature:
                raise ValueError(
                    "Query embedding model does not match the indexed model"
                )
            # Encoding needs no corpus snapshot. A peer may publish while it runs;
            # validate compatibility again in the snapshot used for all evidence.
            query = self._vectors([question], write=False)[0]
        with self._transaction():
            if not self.db.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
                return []
            lexical, dense = [], []
            if mode != "dense":
                expression = (" AND " if match == "all" else " OR ").join(
                    '"' + t + '"' for t in terms
                )
                lexical = [
                    (r[0], -r[1])
                    for r in self.db.execute(
                        "SELECT c.id,fts.rank FROM fts JOIN chunks c ON c.rowid=fts.rowid "
                        "JOIN sections s ON s.id=c.section JOIN documents d ON d.id=s.document "
                        "WHERE fts MATCH ?" + clause + " ORDER BY fts.rank LIMIT ?",
                        [expression, *args, candidates],
                    )
                ]
            if mode != "lexical" and self.embedder:
                if self._state("embedding") != self.embedder.signature:
                    raise ValueError(
                        "Query embedding model does not match the indexed model"
                    )
                if self._state("dimension") != str(len(query)):
                    raise ValueError("Embedding dimensions changed; create a new index")
                dense = self._dense_ranking(query, clause, args, candidates, min_cosine)
            order, scores = fuse(lexical, dense, fusion, dense_weight, candidates)
            rows = {}
            for offset in range(0, len(order), 256):
                ids = order[offset : offset + 256]
                placeholders = ",".join("?" for _ in ids)
                for row in self.db.execute(
                    "SELECT c.id,c.section,c.start,c.end,c.context,s.document,s.locator,s.chars,d.metadata,d.fingerprint FROM chunks c "
                    "JOIN sections s ON s.id=c.section JOIN documents d ON d.id=s.document WHERE c.id IN ("
                    + placeholders
                    + ")",
                    ids,
                ):
                    rows[row["id"]] = row
            if self.reranker and order:
                import numpy as np

                from .token_cache import score

                head, tail = order[:rerank_candidates], order[rerank_candidates:]
                ranked = np.asarray(
                    score(
                        self,
                        question,
                        head,
                        [self._passage(rows[cid]) for cid in head],
                    )
                )
                if ranked.shape != (len(head),) or not np.isfinite(ranked).all():
                    raise ValueError("Invalid reranker scores")
                scores = {cid: float(score) for cid, score in zip(head, ranked)}
                head.sort(key=lambda cid: (-scores[cid], cid))
                # Unscored candidates rank below every rescored one.
                floor = min(scores.values()) - 1
                scores.update({cid: floor for cid in tail})
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


def check_search(
    question,
    *,
    k,
    candidates,
    mode,
    min_cosine,
    match,
    context_chars,
    diversify,
    distinct_documents,
    fusion,
    dense_weight,
    rerank_candidates,
    as_of,
):
    """Reject invalid search options before any store work (shared by stores)."""
    if (
        not isinstance(question, str)
        or len(question) > 4000
        or any(
            not isinstance(v, int) or isinstance(v, bool)
            for v in (k, candidates, context_chars)
        )
        or not 1 <= k <= candidates <= 1000
        or mode not in {"lexical", "dense", "hybrid"}
        or match not in {"all", "any"}
        or not 100 <= context_chars <= 20000
        or not math.isfinite(min_cosine)
        or not -1 <= min_cosine <= 1
        or not isinstance(diversify, bool)
        or not isinstance(distinct_documents, bool)
        or fusion not in {"rrf", "relative"}
        or not isinstance(dense_weight, (int, float))
        or isinstance(dense_weight, bool)
        or not 0 <= dense_weight <= 1
        or not isinstance(rerank_candidates, int)
        or isinstance(rerank_candidates, bool)
        or not 1 <= rerank_candidates <= candidates
    ):
        raise ValueError("Invalid search parameters")
    if as_of is not None and (
        not isinstance(as_of, str) or not _ISO_DATE.fullmatch(as_of)
    ):
        raise ValueError("as_of must be an ISO date or datetime")


def query_terms(question):
    """Distinct lowercase query terms (+, # and _ kept), stop words dropped
    unless the question is only stop words; at most 64."""
    tokens = [
        t
        for t in re.findall(r"[\w+#]+", question.lower())
        if any(c.isalnum() for c in t)
    ]
    terms = list(dict.fromkeys(t for t in tokens if t not in STOP))[:64]
    return terms or list(dict.fromkeys(tokens))[:64]


def fuse(lexical, dense, fusion, dense_weight, candidates):
    """Fused chunk order and scores from (id, score) rankings, best first.

    RRF sums 2*w/(60+rank) per channel (w=0.5 is the classic unweighted sum);
    "relative" sums min-max normalized scores. Ties break on chunk id.
    """
    scores = {}
    for ranking, weight in ((lexical, 1 - dense_weight), (dense, dense_weight)):
        if not ranking or not weight:
            continue
        if fusion == "rrf":
            for rank, (cid, _) in enumerate(ranking, 1):
                scores[cid] = scores.get(cid, 0) + 2 * weight / (60 + rank)
        else:
            low, high = ranking[-1][1], ranking[0][1]
            for cid, value in ranking:
                relative = (value - low) / (high - low) if high > low else 1
                scores[cid] = scores.get(cid, 0) + weight * relative
    return sorted(scores, key=lambda c: (-scores[c], c))[:candidates], scores


def windows(
    order, rows, scores, text, *, k, context_chars, diversify, distinct_documents
):
    """Evidence windows of original text around ranked chunks.

    ``rows`` maps chunk id to its section, offsets, context, document,
    locator, section length, metadata JSON and fingerprint; ``text(section,
    start, end)`` reads original text. Overlapping windows in one section are
    suppressed, not just repeated seed chunks.
    """
    selected, hits, documents = [], [], set()
    for cid in order:
        row = rows[cid]
        if distinct_documents and row["document"] in documents:
            continue
        chunk_end = min(row["end"], row["start"] + context_chars)
        extra = max(0, context_chars - (chunk_end - row["start"])) // 2
        start, end = (
            max(0, row["start"] - extra),
            min(row["chars"], chunk_end + extra),
        )
        if diversify and any(
            section == row["section"]
            and min(previous_end, end) - max(previous_start, start)
            > 0.5 * min(previous_end - previous_start, end - start)
            for section, previous_start, previous_end in selected
        ):
            continue
        selected.append((row["section"], start, end))
        documents.add(row["document"])
        guarded_start, guarded_end = max(0, start - 100), min(row["chars"], end + 100)
        guarded = text(row["section"], guarded_start, guarded_end)
        metadata = row["metadata"]
        hits.append(
            Hit(
                cid,
                row["document"],
                row["locator"],
                start,
                end,
                guarded[start - guarded_start : end - guarded_start],
                json.loads(metadata) if isinstance(metadata, str) else dict(metadata),
                scores[cid],
                row["context"],
                row["fingerprint"],
                guarded[: start - guarded_start],
                guarded[end - guarded_start :],
            )
        )
        if len(hits) == k:
            break
    return hits
