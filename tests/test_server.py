import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from rag_pipeline.server import Engine, parse_keys, serve

ACME, GLOBEX, ADMIN = (
    "acme-key-0123456789",
    "globex-key-012345678",
    "admin-key-0123456789",
)


@pytest.fixture
def api(tmp_path):
    keys = parse_keys(f"acme:{ACME},globex:{GLOBEX},acme:{ADMIN}:admin")
    engine = Engine(tmp_path / "store", keys)
    server = serve(engine, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, key=None, body=None):
        request = Request(
            base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )
        try:
            with urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    yield call, tmp_path
    server.shutdown()
    thread.join()
    server.server_close()


def test_tenants_are_isolated_by_the_store(api):
    call, tmp_path = api
    corpus = tmp_path / "acme-docs"
    corpus.mkdir()
    (corpus / "report.txt").write_text("Acme revenue was USD 125.0 million in 2024.")
    assert call("/v1/ingest", ACME, {"directory": str(corpus)})[0] == 403
    status, stats = call("/v1/ingest", ADMIN, {"directory": str(corpus)})
    assert status == 200 and stats["documents"] == 1
    status, found = call("/v1/search", ACME, {"question": "revenue", "k": 3})
    assert status == 200 and found["hits"][0]["document"] == "report.txt"
    # Globex has its own index file and sees nothing of Acme's.
    status, other = call("/v1/search", GLOBEX, {"question": "revenue"})
    assert status == 200 and other["hits"] == []
    assert (tmp_path / "store" / "acme.sqlite").exists()
    assert (tmp_path / "store" / "globex.sqlite").exists()
    status, answer = call("/v1/ask", ACME, {"question": "revenue"})
    assert status == 200 and answer["status"] == "evidence"


def test_auth_validation_and_audit(api):
    call, tmp_path = api
    assert call("/v1/status")[0] == 401
    assert call("/v1/status", "wrong-key-0123456789")[0] == 401
    assert call("/v1/search", ACME, {"question": "x", "drop": 1})[0] == 400
    assert call("/v1/search", ACME, {"question": "x", "k": 0})[0] == 400
    assert call("/v1/nothing", ACME, {})[0] == 404
    records = [
        json.loads(line)
        for line in (tmp_path / "store" / "audit.jsonl").read_text().splitlines()
    ]
    assert [r["status"] for r in records if "status" in r] == [401, 401, 400, 400, 404]
    assert all("question" not in r for r in records)


@pytest.mark.parametrize(
    "spec",
    ["", "acme", "acme:short", "ac/me:0123456789abcdef", "a:0123456789abcdef:root"],
)
def test_key_specs_are_validated(spec):
    with pytest.raises(ValueError):
        parse_keys(spec)
