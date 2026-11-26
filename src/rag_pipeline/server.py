"""Long-running HTTP server: warm models, many callers, one index per tenant.

    RAG_API_KEYS="acme:k1,globex:k2" rag-pipeline --profile auto serve --root /srv/rag

Each tenant gets its own SQLite index under ``root``, so isolation is enforced
by the store (a key can only open its own file), never by remembering a filter
per query. Requests and their outcome go to an append-only JSON-lines audit
log. Models load once and stay warm across requests.

Endpoints (JSON bodies, ``Authorization: Bearer <key>``):

- ``GET  /v1/status``: counts, generation, plan
- ``POST /v1/search``: ``{"question", "filters"?, "k"?, ...search options}``
- ``POST /v1/ask``: same, plus generation when a chat model is configured
- ``POST /v1/ingest``: ``{"directory"}``, admin keys only
"""

from __future__ import annotations

import hmac
import json
import threading
import time
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .index import Index

SEARCH_OPTIONS = {
    "k",
    "candidates",
    "mode",
    "match",
    "min_cosine",
    "context_chars",
    "diversify",
    "distinct_documents",
    "fusion",
    "dense_weight",
    "rerank_candidates",
    "as_of",
}
MAX_BODY = 1024 * 1024


def parse_keys(spec: str):
    """``tenant:key[:admin],...`` into {key: (tenant, admin)}; tenants are slugs."""
    keys = {}
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        tenant, _, rest = entry.partition(":")
        key, _, role = rest.partition(":")
        if (
            not tenant.replace("-", "").replace("_", "").isalnum()
            or len(key) < 16
            or role not in {"", "admin"}
        ):
            raise ValueError(
                "RAG_API_KEYS entries are tenant:key[:admin]; keys need 16+ characters"
            )
        keys[key] = (tenant, role == "admin")
    if not keys:
        raise ValueError("Set RAG_API_KEYS before serving")
    return keys


class Engine:
    """Tenant-scoped indexes sharing one set of warm providers."""

    def __init__(
        self,
        root,
        keys,
        *,
        embedder=None,
        reranker=None,
        generator=None,
        search=None,
        plan=None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.keys, self.plan = keys, plan
        self.embedder, self.reranker, self.generator = embedder, reranker, generator
        self.search_defaults = dict(search or {})
        self._indexes, self._lock = {}, threading.Lock()
        self._audit = open(self.root / "audit.jsonl", "a", encoding="utf-8")

    def authenticate(self, header):
        scheme, _, token = (header or "").partition(" ")
        if scheme != "Bearer":
            return None
        # Compare against every key in constant time per key.
        match = None
        for key, identity in self.keys.items():
            if hmac.compare_digest(key.encode(), token.encode()):
                match = identity
        return match

    def index(self, tenant):
        with self._lock:
            if tenant not in self._indexes:
                # Server threads share the connection; every use below holds
                # the tenant's lock, so calls never interleave.
                index = Index(
                    self.root / f"{tenant}.sqlite",
                    self.embedder,
                    self.reranker,
                    shared_threads=True,
                )
                self._indexes[tenant] = (index, threading.Lock())
            return self._indexes[tenant]

    def audit(self, **record):
        with self._lock:
            self._audit.write(json.dumps({"time": time.time(), **record}) + "\n")
            self._audit.flush()

    def handle(self, tenant, admin, method, path, body):
        index, lock = self.index(tenant)
        if method == "GET" and path == "/v1/status":
            with lock:
                return 200, index.status()
        if method != "POST":
            return 404, {"error": "Not found"}
        if path == "/v1/ingest":
            if not admin:
                return 403, {"error": "Ingest requires an admin key"}
            with lock:
                return 200, index.ingest(body["directory"], plan=self.plan)
        if path not in {"/v1/search", "/v1/ask"}:
            return 404, {"error": "Not found"}
        question = body.get("question")
        options = {k: v for k, v in body.items() if k in SEARCH_OPTIONS}
        unknown = set(body) - SEARCH_OPTIONS - {"question", "filters"}
        if unknown:
            return 400, {"error": f"Unknown fields: {sorted(unknown)}"}
        options = {**self.search_defaults, **options}
        with lock:
            if path == "/v1/search":
                hits = index.search(question, filters=body.get("filters"), **options)
                return 200, {"hits": [hit.to_dict() for hit in hits]}
            from .answers import answer

            result = answer(
                index,
                question,
                filters=body.get("filters"),
                generate=self.generator,
                **options,
            )
            return 200, asdict(result) | {"tiers": self.plan}


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, payload):
            raw = json.dumps(payload, ensure_ascii=False, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _serve(self, method):
            request_id = uuid.uuid4().hex
            identity = engine.authenticate(self.headers.get("Authorization"))
            if identity is None:
                engine.audit(id=request_id, path=self.path, status=401)
                return self._reply(401, {"error": "Unauthorized"})
            tenant, admin = identity
            started = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    raise ValueError("Request body exceeds 1 MiB")
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                if not isinstance(body, dict):
                    raise ValueError("Request body must be a JSON object")
                status, payload = engine.handle(tenant, admin, method, self.path, body)
            except (ValueError, KeyError, TypeError) as exc:
                status, payload = 400, {"error": f"{type(exc).__name__}: {exc}"}
            except Exception as exc:  # noqa: BLE001 - logged, never leaked
                status, payload = 500, {"error": "Internal error", "id": request_id}
                engine.audit(id=request_id, error=type(exc).__name__)
            engine.audit(
                id=request_id,
                tenant=tenant,
                path=self.path,
                status=status,
                ms=round((time.perf_counter() - started) * 1000, 1),
            )
            self._reply(status, payload)

        def do_GET(self):
            self._serve("GET")

        def do_POST(self):
            self._serve("POST")

        def log_message(self, *args):
            pass

    return Handler


def serve(engine: Engine, host="127.0.0.1", port=8080):
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    return server
