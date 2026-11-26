"""Hosted adapters against a local server that records requests (no vendor calls)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from rag_pipeline.configuration import provider
from rag_pipeline.hosted import AnthropicChat, GeminiEmbedding
from rag_pipeline.providers import EmbeddingServer, RerankingServer


@pytest.fixture
def fake():
    seen = []
    replies = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            headers = {k.lower(): v for k, v in self.headers.items()}
            seen.append((self.path, headers, body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(replies[self.path](body)).encode())

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{http.server_port}", seen, replies
    http.shutdown()
    thread.join()
    http.server_close()


def test_voyage_shape_sends_input_type_per_side(fake, monkeypatch):
    endpoint, seen, replies = fake
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    replies["/v1/embeddings"] = lambda body: {
        "data": [{"index": i, "embedding": [1.0, i]} for i in range(len(body["input"]))]
    }
    model = EmbeddingServer(
        endpoint + "/v1",
        model="voyage-4-large",
        revision="voyage-4-large",
        api_key_env="VOYAGE_API_KEY",
        query_options={"input_type": "query", "output_dimension": 1024},
        passage_options={"input_type": "document", "output_dimension": 1024},
    )
    model.encode(["a", "b"])
    model.encode_queries(["q"])
    assert seen[0][2]["input_type"] == "document"
    assert seen[1][2]["input_type"] == "query"
    assert seen[0][1]["authorization"] == "Bearer k"
    plain = EmbeddingServer(
        endpoint + "/v1", model="voyage-4-large", revision="voyage-4-large"
    )
    assert model.signature != plain.signature
    with pytest.raises(ValueError, match="reserved"):
        EmbeddingServer(endpoint, model="m", revision="r", query_options={"input": []})


@pytest.mark.parametrize(
    "prefix, route",
    [("", "/v2/rerank"), ("/v1", "/models/rerank")],  # Cohere, ZeroEntropy
)
def test_rerank_routes_for_cohere_and_zeroentropy(fake, prefix, route):
    endpoint, seen, replies = fake
    replies[prefix + route] = lambda body: {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]
    }
    model = RerankingServer(
        endpoint + prefix, model="zerank-2", revision="zerank-2", route=route
    )
    assert model.score("q", ["a", "b"]) == [0.1, 0.9]
    assert seen[0][0] == prefix + route and seen[0][2]["documents"] == ["a", "b"]
    with pytest.raises(ValueError, match="route"):
        RerankingServer(endpoint, model="m", revision="r", route="rerank")


def test_gemini_embedding_uses_task_types_and_header_key(fake, monkeypatch):
    endpoint, seen, replies = fake
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    replies["/models/gemini-embedding-001:batchEmbedContents"] = lambda body: {
        "embeddings": [{"values": [0.5] * 64} for _ in body["requests"]]
    }
    model = GeminiEmbedding(dimensions=64, endpoint=endpoint)
    assert len(model.encode(["a", "b"])) == 2
    model.encode_queries(["q"])
    path, headers, body = seen[0]
    assert "secret" not in path and headers["x-goog-api-key"] == "secret"
    assert body["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert seen[1][2]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
    replies["/models/gemini-embedding-001:batchEmbedContents"] = lambda body: {
        "embeddings": [{"values": [0.5] * 3}]
    }
    with pytest.raises(ValueError, match="Invalid Gemini"):
        model.encode(["a"])


class Messages:
    def __init__(self, stop_reason="end_turn", text='{"claims": []}'):
        self.calls, self.stop_reason, self.text = [], stop_reason, text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=self.text)],
        )


def test_anthropic_chat_enables_fallbacks_and_surfaces_refusals():
    messages = Messages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    chat = AnthropicChat(client=client)
    assert chat("system", "evidence") == '{"claims": []}'
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "high"}
    chat.generate_structured("s", "e", {"type": "object"})
    assert messages.calls[1]["output_config"]["format"]["type"] == "json_schema"
    messages.stop_reason = "refusal"
    with pytest.raises(ValueError, match="declined"):
        chat("system", "evidence")


def test_components_build_through_the_provider_factory(monkeypatch):
    from rag_pipeline.capabilities import Capabilities
    from rag_pipeline.components import resolve

    caps = Capabilities(
        cpus=4,
        memory_bytes=8 * 1024**3,
        services=("voyage", "cohere"),
        network=True,
        extras=("models",),
    )
    settings = resolve("workstation", caps).config()
    encoder = provider(settings["embedding"], "embedding")
    reranker = provider(settings["reranker"], "reranker")
    assert encoder.passage_options["input_type"] == "document"
    assert reranker.route == "/v2/rerank"
