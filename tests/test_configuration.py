import json
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from rag_pipeline import Index, answer
from rag_pipeline.configuration import load_config, provider
from rag_pipeline.providers import EmbeddingServer, RerankingServer


@pytest.mark.parametrize(
    "raw",
    [
        '{"embedding":{},"embedding":{}}',
        '{"search":{"bogus":1}}',
        '{"answer":{"max_evidence_chars":NaN}}',
        '{"generation":[]}',
    ],
)
def test_configuration_rejects_ambiguous_or_ignored_settings(tmp_path, raw):
    path = tmp_path / "pipeline.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_config(path)


def test_templates_are_explicit_and_define_representation_identity():
    plain = EmbeddingServer(
        "http://localhost/v1", model="arbitrary-model", revision="release-2026-10-03"
    )
    asymmetric = EmbeddingServer(
        "http://localhost/v1",
        model="arbitrary-model",
        revision="release-2026-10-03",
        query_template="query: {text}",
        passage_template="passage: {text}",
    )
    assert plain.query_template == plain.passage_template == "{text}"
    assert plain.signature != asymmetric.signature
    for template in ["{text.__class__}", "{text}{text}", "{missing}", "{text!r}"]:
        with pytest.raises(ValueError):
            EmbeddingServer(
                "http://localhost/v1",
                model="model",
                revision="v1",
                query_template=template,
            )
    assert callable(
        provider(
            dict(
                kind="python",
                factory="rag_pipeline.providers:ChatServer",
                options=dict(endpoint="http://localhost/v1", model="custom"),
            ),
            "generation",
        )
    )
    with pytest.raises(ValueError, match="Unsupported"):
        provider(dict(kind="onnx", options={}), "generation")


def test_configured_servers_complete_pipeline_and_reject_forged_answer(tmp_path):
    calls = []
    forged = [False]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, data))
            if self.path.endswith("/embeddings"):
                result = dict(
                    data=[
                        dict(
                            index=i,
                            embedding=[1.0, 0.0]
                            if "revenue" in text.lower()
                            else [0.0, 1.0],
                        )
                        for i, text in reversed(list(enumerate(data["input"])))
                    ]
                )
            elif self.path.endswith("/rerank"):
                result = dict(
                    results=[
                        dict(
                            index=i,
                            relevance_score=1.0 if "revenue" in t.lower() else 0.0,
                        )
                        for i, t in reversed(list(enumerate(data["documents"])))
                    ]
                )
            else:
                evidence = json.loads(data["messages"][1]["content"])["evidence"][0]
                claim = dict(
                    text=evidence["text"],
                    evidence=[
                        dict(
                            source_id="forged" if forged[0] else evidence["id"],
                            quote=evidence["text"],
                        )
                    ],
                )
                result = dict(
                    choices=[
                        dict(message=dict(content=json.dumps(dict(claims=[claim]))))
                    ]
                )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        specs = {
            role: dict(
                kind="openai",
                options=dict(
                    endpoint=endpoint,
                    model=f"independent-{role}",
                    **({"revision": "release-1"} if role != "generation" else {}),
                ),
            )
            for role in ("embedding", "reranker", "generation")
        }
        config_path = tmp_path / "pipeline.json"
        specs["search"] = dict(mode="hybrid", k=1, candidates=4)
        config_path.write_text(json.dumps(specs))
        config = load_config(config_path)
        root = tmp_path / "corpus"
        root.mkdir()
        (root / "report.txt").write_text("Revenue was USD 125 million.")
        (root / "legal.txt").write_text("London jurisdiction applies.")
        with Index(
            tmp_path / "index.sqlite",
            provider(config["embedding"], "embedding"),
            provider(config["reranker"], "reranker"),
        ) as index:
            index.ingest(root)
            result = answer(
                index,
                "revenue",
                mode="hybrid",
                generate=provider(config["generation"], "generation"),
            )
            assert (
                result.status == "cited"
                and result.answer == "Revenue was USD 125 million."
            )
            forged[0] = True
            assert (
                answer(
                    index,
                    "revenue",
                    mode="hybrid",
                    generate=provider(config["generation"], "generation"),
                ).status
                == "generation_failed"
            )
        forged[0] = False
        for arguments in [
            ("ingest", str(root)),
            ("search", "revenue"),
            ("ask", "revenue", "--generate"),
        ]:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rag_pipeline.cli",
                    "--config",
                    str(config_path),
                    "--index",
                    str(tmp_path / "cli.sqlite"),
                    *arguments,
                ],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr
            result = json.loads(proc.stdout)
            if arguments[0] == "search":
                assert len(result) == 1 and result[0]["document"] == "report.txt"
            elif arguments[0] == "ask":
                assert result["status"] == "cited"
        assert {data["model"] for _, data in calls} == {
            "independent-embedding",
            "independent-reranker",
            "independent-generation",
        }
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


@pytest.mark.parametrize("commit_peer", [False, True])
def test_ingest_inference_releases_live_writer_and_detects_conflicts(
    tmp_path, commit_peer
):
    path = tmp_path / "index.sqlite"
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "book.md").write_text("# Title\nRevenue evidence.")

    class Model:
        signature = "concurrency-test:v1"

        def encode(self, texts):
            with sqlite3.connect(path, timeout=0.1) as peer:
                peer.execute("BEGIN IMMEDIATE")
                if commit_peer:
                    peer.execute("INSERT INTO state VALUES('peer','committed')")
                else:
                    peer.rollback()
            return np.ones((len(texts), 2), dtype=np.float32)

    with Index(path, Model()) as index:
        if commit_peer:
            with pytest.raises(ValueError, match="Index changed"):
                index.ingest(root, contextual=True)
            assert index.status()["documents"] == 0
            assert index._state("peer") == "committed"
        else:
            assert index.ingest(root, contextual=True)["changed"] == 1
            assert index.search("revenue", mode="hybrid")[0].metadata["title"] == "book"
            assert (
                index.db.execute("SELECT search_text FROM chunks")
                .fetchone()[0]
                .startswith("book\nTitle\n")
            )


def test_reranker_requires_all_candidate_scores():
    model = RerankingServer("http://localhost/v1", model="custom", revision="v1")
    model._request = lambda *a: dict(results=[dict(index=0, relevance_score=1)])
    with pytest.raises(ValueError, match="Reranking server request failed"):
        model.score("query", ["first", "second"])
