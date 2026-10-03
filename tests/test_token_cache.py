import numpy as np
import pytest

from rag_pipeline import Index


class TokenRanker:
    signature = "token-test:v1"

    def __init__(self):
        self.calls = 0

    def encode_query(self, question):
        return np.array([[1.0, 0.0]], dtype=np.float32)

    def encode_documents(self, texts):
        self.calls += len(texts)
        return [
            np.array([[1.0, 0.0] if "100" in t else [0.0, 1.0]], dtype=np.float32)
            for t in texts
        ]

    def score_vectors(self, query, vectors):
        return np.array([(query @ v.T).max(1).sum() for v in vectors])

    def score(self, question, texts):
        return self.score_vectors(
            self.encode_query(question), self.encode_documents(texts)
        )


def test_token_cache_persists_same_rankings_prunes_and_refreshes(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text("Revenue 100.")
    (root / "b.txt").write_text("Revenue 200.")
    model = TokenRanker()
    with Index(tmp_path / "i.sqlite", reranker=model) as index:
        index.ingest(root)
        before = [(h.id, h.score) for h in index.search("revenue", mode="lexical")]
        assert index.prepare_reranker_cache()["prepared"] == 2
        model.calls = 0
        assert [
            (h.id, h.score) for h in index.search("revenue", mode="lexical")
        ] == before
        assert model.calls == 0
        assert index.prepare_reranker_cache()["skipped"] == 2
    with Index(tmp_path / "i.sqlite", reranker=model) as index:
        assert [
            h.document for h in index.search("revenue", mode="lexical", filters={})
        ] == ["a.txt", "b.txt"]
        assert model.calls == 0
        (root / "a.txt").write_text("Revenue 300.")
        (root / "b.txt").unlink()
        index.ingest(root)
        assert index.db.execute("SELECT COUNT(*) FROM rerank_tokens").fetchone()[0] == 0
        assert index.prepare_reranker_cache()["prepared"] == 1
        index.db.execute("UPDATE sections SET text='Revenue 100.'")
        model.calls = 0
        assert index.search("revenue", mode="lexical")[0].score == 1.0
        assert model.calls == 1  # cached source hash no longer agrees
        assert index.prepare_reranker_cache()["prepared"] == 1


def test_token_cache_build_failure_rolls_back_and_corruption_fails(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text("Revenue 100.")
    model = TokenRanker()
    with Index(tmp_path / "i.sqlite", reranker=model) as index:
        index.ingest(root)
        index.prepare_reranker_cache()
        previous = tuple(index.db.execute("SELECT * FROM rerank_tokens").fetchone())
        model.signature = "token-test:v2"
        model.encode_documents = lambda texts: [
            np.array([[float("nan"), 0]]) for t in texts
        ]
        with pytest.raises(ValueError, match="token vectors"):
            index.prepare_reranker_cache()
        assert (
            tuple(index.db.execute("SELECT * FROM rerank_tokens").fetchone())
            == previous
        )
        model.signature = "token-test:v1"
        index.db.execute(
            "UPDATE rerank_tokens SET vector=?",
            (np.array([[0.0, 0.0]], dtype="<f4").tobytes(),),
        )
        with pytest.raises(ValueError, match="token vectors"):
            index.search("revenue", mode="lexical")
