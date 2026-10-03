"""Optional self-hosted embedding protocol, independent of model framework.

Compatible with OpenAI-style /embeddings servers (including llama.cpp/vLLM).
Model execution stays outside the transactional store. Callers must configure
an immutable revision/digest matching the server's actual model artifact.
"""

from __future__ import annotations

import json
import math
import os
import re
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Embedding endpoint redirects are not supported")


class EmbeddingServer:
    def __init__(
        self,
        endpoint,
        *,
        model,
        revision,
        instruction="Given a question, retrieve relevant passages that answer the question.",
        api_key_env=None,
        timeout=120,
        batch_size=8,
    ):
        if not isinstance(endpoint, str):
            raise ValueError("Invalid embedding endpoint")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Invalid embedding endpoint")
        if (
            any(
                not isinstance(x, str) or not x.strip() or len(x) > 512
                for x in (model, revision)
            )
            or not isinstance(instruction, str)
            or len(instruction) > 2000
        ):
            raise ValueError("Model and immutable revision are required")
        if not re.fullmatch(r"(?:[0-9a-f]{40}|(?:sha256:)?[0-9a-f]{64})", revision):
            raise ValueError("Embedding revision must be a commit or SHA-256 digest")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 64
            or not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or not 1 <= timeout <= 600
        ):
            raise ValueError("Invalid embedding request limits")
        self.endpoint, self.model = endpoint.rstrip("/"), model
        self.instruction, self.api_key_env = instruction, api_key_env
        self.timeout, self.batch_size = timeout, batch_size
        # Endpoint/credentials deliberately do not define model compatibility.
        self.signature = json.dumps(
            dict(
                model=model,
                revision=revision,
                instruction=instruction,
                protocol="openai-embeddings:qwen-instruct:v1",
            ),
            sort_keys=True,
        )

    def encode_queries(self, texts):
        return self.encode(
            [
                f"Instruct: {self.instruction}\nQuery: {t}" if self.instruction else t
                for t in texts
            ]
        )

    def encode(self, texts):
        vectors = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            if any(not isinstance(t, str) for t in batch):
                raise ValueError("Embedding inputs must be strings")
            payload = json.dumps(dict(model=self.model, input=batch)).encode()
            if len(payload) > 256000:
                raise ValueError("Embedding request exceeds 256KB")
            headers = {"Content-Type": "application/json"}
            if self.api_key_env:
                key = os.environ.get(self.api_key_env)
                if not key:
                    raise ValueError("Embedding API credential is missing")
                headers["Authorization"] = "Bearer " + key
            try:
                request = Request(
                    self.endpoint + "/embeddings", data=payload, headers=headers
                )
                with build_opener(_NoRedirect()).open(
                    request, timeout=self.timeout
                ) as response:
                    raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError("Oversized embedding response")
                data = json.loads(raw)["data"]
                if not isinstance(data, list) or len(data) != len(batch):
                    raise ValueError("Invalid embedding batch")
                ordered = [None] * len(batch)
                for row in data:
                    position, vector = row["index"], row["embedding"]
                    if (
                        not isinstance(position, int)
                        or isinstance(position, bool)
                        or not 0 <= position < len(batch)
                        or ordered[position] is not None
                    ):
                        raise ValueError("Invalid embedding response index")
                    if (
                        not isinstance(vector, list)
                        or not 1 <= len(vector) <= 65536
                        or any(
                            not isinstance(v, (int, float))
                            or isinstance(v, bool)
                            or not math.isfinite(v)
                            for v in vector
                        )
                    ):
                        raise ValueError("Invalid embedding vector")
                    ordered[position] = vector
                vectors.extend(ordered)
            except Exception:
                # Never surface server bodies containing prompts or credentials.
                raise ValueError("Embedding server request failed") from None
        return vectors
