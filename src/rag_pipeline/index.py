"""One transactional source of truth for text, filters, lexical and dense retrieval."""

from __future__ import annotations

import heapq
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .sources import FORMATS, Document, document, metadata, spans

SCHEMA = """
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, metadata TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS document_metadata(document TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(document,key));
CREATE INDEX IF NOT EXISTS metadata_lookup ON document_metadata(key,value,document);
CREATE TABLE IF NOT EXISTS sections(id INTEGER PRIMARY KEY, document TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, locator TEXT NOT NULL, text TEXT NOT NULL, chars INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(id TEXT UNIQUE NOT NULL, section INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE, start INTEGER NOT NULL, end INTEGER NOT NULL, search_text TEXT NOT NULL, vector BLOB);
CREATE INDEX IF NOT EXISTS section_document ON sections(document);
CREATE INDEX IF NOT EXISTS chunk_section ON chunks(section);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(search_text, content=chunks, content_rowid=rowid, tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS chunk_insert AFTER INSERT ON chunks BEGIN
 INSERT INTO fts(rowid,search_text) VALUES(new.rowid,new.search_text); END;
CREATE TRIGGER IF NOT EXISTS chunk_delete AFTER DELETE ON chunks BEGIN
 INSERT INTO fts(fts,rowid,search_text) VALUES('delete',old.rowid,old.search_text); END;
"""
STOP = set(
    "a an and are as at be by can do does for from how i in is it of on or that the their this to was were what when where which who why with would".split()
)


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

    def to_dict(self):
        return asdict(self)


class Index:
    def __init__(self, path: str | Path, embedder=None, reranker=None):
        self.path = Path(path)
        existed = self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.embedder = embedder
        self.reranker = reranker
        if embedder is not None and not getattr(embedder, "signature", "").strip():
            self.db.close()
            raise ValueError("Embedder requires a stable model/revision signature")
        try:
            if existed and self._state("schema") != "2":
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
            if version not in (None, "2"):
                raise ValueError("Unsupported index schema; create a new index")
            self.db.execute("INSERT OR IGNORE INTO state VALUES('schema','2')")
        except Exception:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.db.close()

    def _state(self, key):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    @contextmanager
    def _transaction(self, write=False):
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

    def ingest(self, root: str | Path, *, size=900, overlap=100):
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(
                "Source directory must exist; refusing to prune an absent corpus"
            )
        # Validate before a no-op ingest as well.
        list(spans("", size, overlap))
        paths = sorted(
            p
            for p in root.rglob("*")
            if p.suffix.lower() in FORMATS
            and not p.name.endswith(".meta.json")
            and p.is_file()
        )
        if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in paths):
            raise ValueError("Symlinked sources are not supported")
        signature = getattr(self.embedder, "signature", "")
        pipeline = json.dumps(["extract-v1", size, overlap, signature])
        with self._transaction(write=True):
            if self._state("embedding") and not self.embedder:
                raise ValueError(
                    "This index has vectors; supply the same embedder to ingest"
                )
            rebuild = self._state("pipeline") != pipeline
            if self._state("embedding") != signature:
                self.db.execute("DELETE FROM state WHERE key='dimension'")
            existing = dict(self.db.execute("SELECT id,fingerprint FROM documents"))
            seen, changed = set(), 0
            for path in paths:
                doc_id = path.relative_to(root).as_posix()
                doc = document(path, root, None if rebuild else existing.get(doc_id))
                seen.add(doc.id)
                if not rebuild and existing.get(doc.id) == doc.fingerprint:
                    continue
                self._replace(doc, size, overlap)
                changed += 1
            removed = existing.keys() - seen
            self.db.executemany(
                "DELETE FROM documents WHERE id=?", [(i,) for i in removed]
            )
            for key, value in [("pipeline", pipeline), ("embedding", signature)]:
                self.db.execute(
                    "INSERT OR REPLACE INTO state VALUES(?,?)", (key, value)
                )
            if changed or removed:
                self.db.execute(
                    "INSERT OR REPLACE INTO state VALUES(?,?)",
                    ("generation", str(int(self._state("generation") or 0) + 1)),
                )
        return {"changed": changed, "removed": len(removed), **self.status()}

    def _replace(self, doc: Document, size: int, overlap: int):
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
            offsets = list(spans(text, size, overlap))
            for offset in range(0, len(offsets), 32):
                batch = offsets[offset : offset + 32]
                texts = [title[:300] + "\n" + text[a:b] for a, b in batch]
                vectors = self._vectors(texts) if self.embedder else [None] * len(batch)
                self.db.executemany(
                    "INSERT INTO chunks VALUES(?,?,?,?,?,?)",
                    [
                        (
                            f"{doc.id}#{locator}:{a}-{b}",
                            section,
                            a,
                            b,
                            t,
                            v.tobytes() if v is not None else None,
                        )
                        for (a, b), t, v in zip(batch, texts, vectors)
                    ],
                )

    def _text(self, section, start, end):
        return self.db.execute(
            "SELECT substr(text,?,?) FROM sections WHERE id=?",
            (start + 1, end - start, section),
        ).fetchone()[0]

    def _vectors(self, texts, write=True):
        import numpy as np

        values = np.asarray(self.embedder.encode(texts), dtype="<f4")
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
        mode="hybrid",
        min_cosine=0.3,
        match="any",
        context_chars=1800,
    ) -> list[Hit]:
        if (
            not isinstance(question, str)
            or len(question) > 4000
            or not 1 <= k <= candidates <= 1000
            or mode not in {"lexical", "dense", "hybrid"}
            or match not in {"all", "any"}
            or not 100 <= context_chars <= 20000
            or not math.isfinite(min_cosine)
            or not -1 <= min_cosine <= 1
        ):
            raise ValueError("Invalid search parameters")
        terms = list(
            dict.fromkeys(
                t for t in re.findall(r"\w+", question.lower()) if t not in STOP
            )
        )[:64]
        if not terms:
            return []
        clause, args = self._filters(filters)
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
                import numpy as np

                if self._state("embedding") != self.embedder.signature:
                    raise ValueError(
                        "Query embedding model does not match the indexed model"
                    )
                query = self._vectors([question], write=False)[0]
                cursor = self.db.execute(
                    "SELECT c.id,c.vector FROM chunks c JOIN sections s ON s.id=c.section "
                    "JOIN documents d ON d.id=s.document WHERE c.vector IS NOT NULL"
                    + clause,
                    args,
                )
                best = []
                while rows := cursor.fetchmany(512):
                    matrix = np.stack([np.frombuffer(r[1], dtype="<f4") for r in rows])
                    if matrix.shape[1] != len(query) or not np.isfinite(matrix).all():
                        raise ValueError("Corrupt stored vectors")
                    scores = matrix @ query
                    best = heapq.nsmallest(
                        candidates,
                        best
                        + [
                            (-float(score), r[0])
                            for r, score in zip(rows, scores)
                            if score >= min_cosine
                        ],
                    )
                dense = [cid for _, cid in best]
            if mode == "dense" and not self.embedder:
                raise ValueError("Dense search requires an embedder")
            scores = {}
            for ranking in (lexical, dense):
                for rank, cid in enumerate(ranking, 1):
                    scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
            order = sorted(scores, key=lambda c: (-scores[c], c))[:candidates]
            rows = {
                cid: self.db.execute(
                    "SELECT c.id,c.section,c.start,c.end,s.document,s.locator,s.chars,d.metadata FROM chunks c "
                    "JOIN sections s ON s.id=c.section JOIN documents d ON d.id=s.document WHERE c.id=?",
                    (cid,),
                ).fetchone()
                for cid in order
            }
            if self.reranker and order:
                import numpy as np

                ranked = np.asarray(
                    self.reranker.score(
                        question,
                        [
                            self._text(
                                rows[cid]["section"],
                                rows[cid]["start"],
                                rows[cid]["end"],
                            )
                            for cid in order
                        ],
                    )
                )
                if ranked.shape != (len(order),) or not np.isfinite(ranked).all():
                    raise ValueError("Invalid reranker scores")
                scores = {cid: float(score) for cid, score in zip(order, ranked)}
                order.sort(key=lambda cid: (-scores[cid], cid))
            selected, hits = [], []
            for cid in order:
                row = rows[cid]
                # Suppress heavily overlapping chunks from the same section.
                if any(
                    section == row["section"]
                    and min(end, row["end"]) - max(start, row["start"])
                    > 0.5 * min(end - start, row["end"] - row["start"])
                    for section, start, end in selected
                ):
                    continue
                selected.append((row["section"], row["start"], row["end"]))
                chunk_end = min(row["end"], row["start"] + context_chars)
                extra = max(0, context_chars - (chunk_end - row["start"])) // 2
                start, end = (
                    max(0, row["start"] - extra),
                    min(row["chars"], chunk_end + extra),
                )
                hits.append(
                    Hit(
                        cid,
                        row["document"],
                        row["locator"],
                        start,
                        end,
                        self._text(row["section"], start, end),
                        json.loads(row["metadata"]),
                        scores[cid],
                    )
                )
                if len(hits) == k:
                    break
            return hits
