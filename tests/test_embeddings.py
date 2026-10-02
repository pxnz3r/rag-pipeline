import os

import numpy as np
import pytest

from rag_pipeline.embeddings import CrossEncoder, MiniLM


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

    ranker = CrossEncoder()
    scores = ranker.score(
        "How much did sales increase?",
        [
            "Revenue grew substantially this year.",
            "The courts of London have exclusive jurisdiction.",
        ],
    )
    assert scores.shape == (2,) and np.isfinite(scores).all() and scores[0] > scores[1]
