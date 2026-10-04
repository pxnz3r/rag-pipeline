"""One transactional source of truth for text, filters, lexical and dense retrieval."""

from __future__ import annotations

import heapq
import json
import math
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .sources import (
    FORMATS,
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


def _walk_error(error):
    raise error


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
        vector_cache_bytes=64 * 1024 * 1024,
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
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
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
            }

    def ingest(self, root: str | Path, *, size=900, overlap=100, contextual=False):
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
        pipeline = json.dumps(["extract-v4", size, overlap, signature, contextual])
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
        with tempfile.TemporaryDirectory(prefix="rag-ingest-") as directory:
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
                    for path in paths:
                        doc = document(
                            path,
                            root,
                            None
                            if rebuild
                            else existing.get(path.relative_to(root).as_posix()),
                        )
                        seen.add(doc.id)
                        if not rebuild and existing.get(doc.id) == doc.fingerprint:
                            continue
                        staged._replace(doc, size, overlap, contextual, batch)
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
                    ]:
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
        self, doc: Document, size: int, overlap: int, contextual: bool, batch: list
    ):
        self.db.execute("DELETE FROM documents WHERE id=?", (doc.id,))
        self.db.execute(
            "INSERT INTO documents VALUES(?,?,?)",
            (doc.id, doc.fingerprint, json.dumps(metadata(doc.metadata))),
        )
        self.db.executemany(
            "INSERT INTO document_metadata VALUES(?,?,?)",
            [(doc.id, key, str(value)) for key, value in doc.metadata.items()],
        )
        title = " ".join(str(v) for v in doc.metadata.values())
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
            for a, b, context in contextual_spans(text, size, overlap, contextual):
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
                if len(batch) == 32:
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
        import numpy as np

        def top(scores, ids):
            positions = np.flatnonzero(scores >= min_cosine)
            if len(positions) > candidates:
                cutoff = np.partition(scores[positions], -candidates)[-candidates]
                positions = positions[scores[positions] >= cutoff]
            return heapq.nsmallest(
                candidates, [(-float(scores[i]), ids[i]) for i in positions]
            )

        def unpack(rows):
            block = np.stack([np.frombuffer(r[1], dtype="<f4") for r in rows])
            if (
                block.shape[1] != len(query)
                or not np.isfinite(block).all()
                or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-3)
            ):
                raise ValueError("Corrupt stored vectors")
            return block

        if not self.vector_cache_bytes:
            self._dense_cache = None
        token = (
            self._state("generation"),
            self.db.total_changes,
            self.db.execute("PRAGMA data_version").fetchone()[0],
            len(query),
            self.vector_cache_bytes,
        )
        cached = None
        if not clause and self.vector_cache_bytes:
            if self._dense_cache is None or self._dense_cache[0] != token:
                self._dense_cache = (token, None)
                count, chars = self.db.execute(
                    "SELECT COUNT(*),COALESCE(SUM(length(id)),0) FROM chunks"
                ).fetchone()
                # Conservative allowance for IDs, references, scores and selection
                # temporaries; model/native/SQLite allocations are separate.
                if (
                    count * (4 * len(query) + 256) + 4 * chars
                    <= self.vector_cache_bytes
                ):
                    matrix = np.empty((count, len(query)), dtype="<f4")
                    ids, offset = [], 0
                    cursor = self.db.execute(
                        "SELECT id,vector FROM chunks WHERE vector IS NOT NULL"
                    )
                    while rows := cursor.fetchmany(512):
                        block = unpack(rows)
                        matrix[offset : offset + len(rows)] = block
                        ids.extend(r[0] for r in rows)
                        offset += len(rows)
                    self._dense_cache = (token, (matrix[:offset], ids))
            cached = self._dense_cache[1]
        if cached is not None:
            matrix, ids = cached
            best = top(np.einsum("ij,j->i", matrix, query), ids)
        else:
            cursor = self.db.execute(
                "SELECT c.id,c.vector FROM chunks c JOIN sections s ON s.id=c.section "
                "JOIN documents d ON d.id=s.document WHERE c.vector IS NOT NULL"
                + clause,
                args,
            )
            best = []
            while rows := cursor.fetchmany(512):
                matrix = unpack(rows)
                best = heapq.nsmallest(
                    candidates,
                    best
                    + top(np.einsum("ij,j->i", matrix, query), [r[0] for r in rows]),
                )
        return [cid for _, cid in best]

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
    ) -> list[Hit]:
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
        ):
            raise ValueError("Invalid search parameters")
        if mode in {"dense", "hybrid"} and not self.embedder:
            raise ValueError("Dense/hybrid search requires an embedder")
        terms = list(
            dict.fromkeys(
                t
                for t in re.findall(r"[\w+#]+", question.lower())
                if any(c.isalnum() for c in t) and t not in STOP
            )
        )[:64]
        if not terms:
            terms = list(
                dict.fromkeys(
                    t
                    for t in re.findall(r"[\w+#]+", question.lower())
                    if any(c.isalnum() for c in t)
                )
            )[:64]
        if not terms:
            return []
        clause, args = self._filters(filters)
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
                    r[0]
                    for r in self.db.execute(
                        "SELECT c.id FROM fts JOIN chunks c ON c.rowid=fts.rowid "
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
            scores = {}
            for ranking in (lexical, dense):
                for rank, cid in enumerate(ranking, 1):
                    scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
            order = sorted(scores, key=lambda c: (-scores[c], c))[:candidates]
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

                ranked = np.asarray(
                    score(
                        self,
                        question,
                        order,
                        [self._passage(rows[cid]) for cid in order],
                    )
                )
                if ranked.shape != (len(order),) or not np.isfinite(ranked).all():
                    raise ValueError("Invalid reranker scores")
                scores = {cid: float(score) for cid, score in zip(order, ranked)}
                order.sort(key=lambda cid: (-scores[cid], cid))
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
                # Suppress duplicate evidence windows, not merely seed chunks.
                if diversify and any(
                    section == row["section"]
                    and min(previous_end, end) - max(previous_start, start)
                    > 0.5 * min(previous_end - previous_start, end - start)
                    for section, previous_start, previous_end in selected
                ):
                    continue
                selected.append((row["section"], start, end))
                documents.add(row["document"])
                guarded_start, guarded_end = (
                    max(0, start - 100),
                    min(row["chars"], end + 100),
                )
                guarded = self._text(row["section"], guarded_start, guarded_end)
                hits.append(
                    Hit(
                        cid,
                        row["document"],
                        row["locator"],
                        start,
                        end,
                        guarded[start - guarded_start : end - guarded_start],
                        json.loads(row["metadata"]),
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
