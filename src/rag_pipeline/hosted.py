"""Hosted model adapters that don't fit the OpenAI-shaped servers in ``providers``.

Voyage embeddings and Cohere or ZeroEntropy reranking use ``EmbeddingServer``
and ``RerankingServer`` with vendor options (see ``components``). This module
holds the two with their own protocols: Claude through the official
``anthropic`` SDK, and Gemini's native batch embedding endpoint. API keys come
from the environment only.
"""

from __future__ import annotations

import math

from .providers import _Server, _signature


class AnthropicChat:
    """Answer generation with Claude, returning text in the generator contract.

    The callable matches ``answer``'s ``generate(system, evidence) -> str``.
    Server-side refusal fallbacks are on: a declined request re-runs on the
    fallback model the API routes it to, inside the same call.
    """

    def __init__(
        self,
        model="claude-opus-5-5",
        *,
        max_tokens=16000,
        effort="high",
        timeout=600,
        client=None,
    ):
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Invalid effort level")
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout)
        self.client, self.model = client, model
        self.max_tokens, self.effort = max_tokens, effort

    def __call__(self, system, evidence):
        return self._create(system, evidence)

    def generate_structured(self, system, evidence, schema):
        return self._create(
            system,
            evidence,
            {"type": "json_schema", "schema": schema},
        )

    def _create(self, system, evidence, schema=None):
        output = {"effort": self.effort}
        if schema is not None:
            output["format"] = schema
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": evidence}],
            output_config=output,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise ValueError("The model declined to answer from this evidence")
        if response.stop_reason == "max_tokens":
            raise ValueError("Generation hit max_tokens before finishing")
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text:
            raise ValueError("Empty generation response")
        return text


class GeminiEmbedding(_Server):
    """Gemini ``batchEmbedContents`` with retrieval task types.

    Queries use RETRIEVAL_QUERY and passages RETRIEVAL_DOCUMENT; the model and
    output dimensionality enter the signature.
    """

    def __init__(
        self,
        model="gemini-embedding-001",
        *,
        dimensions=768,
        batch_size=100,
        endpoint="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="GEMINI_API_KEY",
        **kwargs,
    ):
        super().__init__(endpoint, model=model, api_key_env=None, **kwargs)
        if not isinstance(dimensions, int) or not 64 <= dimensions <= 4096:
            raise ValueError("Invalid embedding dimensionality")
        self.key_env, self.dimensions, self.batch_size = (
            api_key_env,
            dimensions,
            batch_size,
        )
        self.signature = _signature(
            model, model, dimensions=dimensions, protocol="gemini-batch-embed:v1"
        )

    def encode(self, texts):
        return self._encode(texts, "RETRIEVAL_DOCUMENT")

    def encode_queries(self, texts):
        return self._encode(texts, "RETRIEVAL_QUERY")

    def _encode(self, texts, task):
        import os

        key = os.environ.get(self.key_env)
        if not key:
            raise ValueError("Gemini API credential is missing")
        vectors = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            requests = [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task,
                    "outputDimensionality": self.dimensions,
                }
                for text in batch
            ]
            data = self._post(
                f"/models/{self.model}:batchEmbedContents", {"requests": requests}, key
            )
            rows = data.get("embeddings")
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise ValueError("Invalid Gemini embedding batch")
            for row in rows:
                values = row.get("values")
                if (
                    not isinstance(values, list)
                    or len(values) != self.dimensions
                    or not all(
                        isinstance(v, (int, float)) and math.isfinite(v) for v in values
                    )
                ):
                    raise ValueError("Invalid Gemini embedding vector")
                vectors.append(values)
        return vectors

    def _post(self, route, payload, key):
        import json
        from urllib.request import Request

        from .providers import _NoRedirect, build_opener

        request = Request(
            self.endpoint + route,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        try:
            with build_opener(_NoRedirect()).open(
                request, timeout=self.timeout
            ) as response:
                raw = response.read(64 * 1024 * 1024 + 1)
            if len(raw) > 64 * 1024 * 1024:
                raise ValueError("Oversized embedding response")
            return json.loads(raw)
        except Exception:
            raise ValueError("Gemini embedding request failed") from None
