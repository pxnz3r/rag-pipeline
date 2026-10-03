"""Transactional passage-token cache for exact candidate MaxSim reranking.

This trades disk/indexing work for query latency; it is not an ANN token index.
Float32 storage preserves the encoder outputs without quantization changes.
"""

from __future__ import annotations

import hashlib

import numpy as np

DDL = """CREATE TABLE IF NOT EXISTS rerank_tokens(
chunk TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
model TEXT NOT NULL, fingerprint TEXT NOT NULL,
tokens INTEGER NOT NULL, dimensions INTEGER NOT NULL, vector BLOB NOT NULL)"""


def _fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _vectors(values):
    matrix = np.asarray(values, dtype="<f4")
    if (
        matrix.ndim != 2
        or not 1 <= matrix.shape[0] <= 512
        or not 1 <= matrix.shape[1] <= 4096
        or not np.isfinite(matrix).all()
        or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3)
    ):
        raise ValueError("Invalid normalized reranker token vectors")
    return matrix


def prepare(index):
    model = index.reranker
    if (
        not all(
            callable(getattr(model, name, None))
            for name in ("encode_documents", "encode_query", "score_vectors")
        )
        or not isinstance(getattr(model, "signature", None), str)
        or not model.signature
    ):
        raise ValueError("Token caching requires a versioned token reranker")
    prepared, skipped = 0, 0
    with index._transaction(write=True):
        index.db.execute(DDL)
        # A single active checkpoint, not an unbounded cache of every past model.
        index.db.execute("DELETE FROM rerank_tokens WHERE model<>?", (model.signature,))
        cursor = index.db.execute(
            "SELECT c.id,c.section,c.start,c.end,c.context,t.fingerprint cached FROM chunks c "
            "LEFT JOIN rerank_tokens t ON t.chunk=c.id ORDER BY c.rowid"
        )
        while rows := cursor.fetchmany(8):
            texts = [index._passage(row) for row in rows]
            pending = [
                (row, text)
                for row, text in zip(rows, texts)
                if row["cached"] != _fingerprint(text)
            ]
            skipped += len(rows) - len(pending)
            if not pending:
                continue
            rows, texts = zip(*pending)
            texts = list(texts)
            encoded = model.encode_documents(texts)
            if len(encoded) != len(rows):
                raise ValueError("Invalid reranker token batch")
            for row, text, vector in zip(rows, texts, encoded):
                matrix = _vectors(vector)
                index.db.execute(
                    "INSERT OR REPLACE INTO rerank_tokens VALUES(?,?,?,?,?,?)",
                    (
                        row["id"],
                        model.signature,
                        _fingerprint(text),
                        *matrix.shape,
                        matrix.tobytes(),
                    ),
                )
                prepared += 1
    return dict(prepared=prepared, skipped=skipped, model=model.signature)


def score(index, question, ids, texts):
    model = index.reranker
    if not all(
        callable(getattr(model, name, None))
        for name in ("encode_documents", "encode_query", "score_vectors")
    ):
        return model.score(question, texts)
    if not index.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rerank_tokens'"
    ).fetchone():
        return model.score(question, texts)
    vectors, missing = [None] * len(ids), []
    for position, (cid, text) in enumerate(zip(ids, texts)):
        row = index.db.execute(
            "SELECT * FROM rerank_tokens WHERE chunk=?", (cid,)
        ).fetchone()
        if (
            row is None
            or row["model"] != model.signature
            or row["fingerprint"] != _fingerprint(text)
        ):
            missing.append(position)
            continue
        if (
            not isinstance(row["tokens"], int)
            or not isinstance(row["dimensions"], int)
            or not isinstance(row["vector"], bytes)
            or not 1 <= row["tokens"] <= 512
            or not 1 <= row["dimensions"] <= 4096
            or len(row["vector"]) != 4 * row["tokens"] * row["dimensions"]
        ):
            raise ValueError("Corrupt cached reranker vectors")
        vectors[position] = _vectors(
            np.frombuffer(row["vector"], dtype="<f4").reshape(
                row["tokens"], row["dimensions"]
            )
        )
    if missing:
        encoded = model.encode_documents([texts[p] for p in missing])
        if len(encoded) != len(missing):
            raise ValueError("Invalid reranker token batch")
        for position, vector in zip(missing, encoded):
            vectors[position] = _vectors(vector)
    query = _vectors(model.encode_query(question))
    if any(v.shape[1] != query.shape[1] for v in vectors):
        raise ValueError("Reranker token dimensions differ")
    return model.score_vectors(query, vectors)
