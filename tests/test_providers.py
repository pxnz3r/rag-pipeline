import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rag_pipeline.providers import EmbeddingServer


@pytest.fixture
def server():
    requests = []
    response = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{http.server_port}/v1", requests, response
    http.shutdown()
    thread.join()
    http.server_close()


def test_embedding_server_reorders_and_separates_query_instruction(server):
    endpoint, requests, _ = server
    model = EmbeddingServer(
        endpoint, model="qwen3", revision="a" * 64, instruction="Retrieve evidence."
    )
    assert model.encode(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert model.encode_queries(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert requests[0]["input"] == ["first", "second"]
    assert requests[1]["input"][0].startswith("Instruct:")
    assert "\nQuery: first" in requests[1]["input"][0]
    assert (
        model.signature
        != EmbeddingServer(endpoint, model="qwen3", revision="b" * 64).signature
    )


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}],
        [{"index": 0, "embedding": [float("nan")]}, {"index": 1, "embedding": [1]}],
        [{"index": True, "embedding": [1]}, {"index": 0, "embedding": [1]}],
        [{"index": 0, "embedding": [True]}, {"index": 1, "embedding": [1]}],
        [{"index": 0, "embedding": []}, {"index": 1, "embedding": [1]}],
    ],
)
def test_embedding_server_rejects_corrupt_protocol_without_response_leaks(server, data):
    endpoint, _, response = server
    response["data"] = data
    with pytest.raises(ValueError, match="^Embedding server request failed$"):
        EmbeddingServer(endpoint, model="qwen3", revision="a" * 64).encode(["a", "b"])


@pytest.mark.parametrize(
    "endpoint,revision",
    [
        ("http://secret:password@example.org/v1", "a" * 64),
        ("file:///tmp/model", "a" * 64),
        ("http://example.org/v1", "main"),
    ],
)
def test_embedding_server_requires_explicit_endpoint_and_revision(endpoint, revision):
    with pytest.raises(ValueError):
        EmbeddingServer(endpoint, model="qwen3", revision=revision)


def test_generation_schema_is_explicit_and_sent_unchanged(server):
    from rag_pipeline import Hit
    from rag_pipeline.providers import ChatServer
    from rag_pipeline.reasoning import program_schema

    endpoint, requests, response = server
    response.clear()
    response["choices"] = [dict(message=dict(content='{"steps":[]}'))]
    text = "Revenue 125 and 100."
    schema = program_schema(
        [Hit("source", "doc", "text", 0, len(text), text, {}, 0)], catalog=True
    )
    client = ChatServer(endpoint, model="operator-model", structured_outputs=True)
    assert client.generate_structured("system", "{}", schema) == '{"steps":[]}'
    assert requests[-1]["response_format"]["json_schema"]["schema"] == schema
    assert requests[-1]["model"] == "operator-model"
    with pytest.raises(ValueError, match="capability"):
        ChatServer(endpoint, model="operator-model").generate_structured(
            "system", "{}", schema
        )
