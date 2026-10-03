import os

import numpy as np
import pytest

from rag_pipeline.embeddings import E5, ColBERT, CrossEncoder, MiniLM, ModernColBERT


@pytest.mark.skipif(
    os.environ.get("RAG_TEST_MODELS") != "1", reason="Opt-in public model download"
)
def test_pinned_cpu_model_semantic_paraphrase():
    model = MiniLM()
    vectors = model.encode(
        [
            "How much did sales increase?",
            "Revenue grew substantially this year.",
            "The courts of London have exclusive jurisdiction.",
        ]
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert vectors.shape == (3, 384) and np.isfinite(vectors).all()
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]

    from rag_pipeline.embeddings import ONNXEmbedding, ONNXReranker

    configured = ONNXEmbedding(
        model=model.model,
        revision=model.revision,
        dimensions=384,
        length=256,
        pooling="mean",
    )
    raw = configured.encode(
        [
            "How much did sales increase?",
            "Revenue grew substantially this year.",
            "The courts of London have exclusive jurisdiction.",
        ]
    )
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    assert np.allclose(raw, vectors, atol=1e-5)
    assert configured.encode([]).shape == (0, 384)
    reversed_vectors = model.encode(
        [
            "The courts of London have exclusive jurisdiction.",
            "Revenue grew substantially this year.",
            "How much did sales increase?",
        ]
    )
    reversed_vectors /= np.linalg.norm(reversed_vectors, axis=1, keepdims=True)
    assert np.allclose(vectors, reversed_vectors[::-1], atol=1e-5)

    ranker = CrossEncoder()
    scores = ranker.score(
        "How much did sales increase?",
        [
            "Revenue grew substantially this year.",
            "The courts of London have exclusive jurisdiction.",
        ],
    )
    assert scores.shape == (2,) and np.isfinite(scores).all() and scores[0] > scores[1]

    configured_ranker = ONNXReranker(
        model=ranker.model, revision=ranker.revision, length=512
    )
    assert np.allclose(
        configured_ranker.score(
            "How much did sales increase?",
            [
                "Revenue grew substantially this year.",
                "The courts of London have exclusive jurisdiction.",
            ],
        ),
        scores,
        atol=1e-5,
    )

    asymmetric = E5()
    passages = asymmetric.encode(
        [
            "Revenue grew substantially this year.",
            "The courts of London have exclusive jurisdiction.",
        ]
    )
    query = asymmetric.encode_queries(["How much did sales increase?"])
    passages /= np.linalg.norm(passages, axis=1, keepdims=True)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    assert passages.shape == (2, 384) and np.isfinite(passages).all()
    assert (passages @ query[0])[0] > (passages @ query[0])[1]

    token_ranker = ColBERT()
    texts = ["Hayao Miyazaki directed Spirited Away.", "Walt Disney founded Disney."]
    scores = token_ranker.score("Who directed Spirited Away?", texts)
    assert scores.shape == (2,) and np.isfinite(scores).all() and scores[0] > scores[1]
    assert np.allclose(
        scores,
        token_ranker.score("Who directed Spirited Away?", texts[::-1])[::-1],
        atol=1e-5,
    )
    assert token_ranker._tokens(texts)[0].shape[1] == 96
    assert token_ranker._tokens(["Who directed Spirited Away?"], query=True)[
        0
    ].shape == (32, 96)
    modern = ModernColBERT()
    modern_scores = modern.score("Who directed Spirited Away?", texts)
    assert modern_scores.shape == (2,) and np.isfinite(modern_scores).all()
    assert modern_scores[0] > modern_scores[1]
    assert modern._tokens(texts)[0].shape[1] == 128
