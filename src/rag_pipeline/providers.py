"""Bounded model-server adapters with explicit model and representation settings."""

from __future__ import annotations

import json
import math
import os
from string import Formatter
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Model endpoint redirects are not supported")


class _Server:
    def __init__(self, endpoint, *, model, api_key_env=None, timeout=120):
        parsed = urlsplit(endpoint) if isinstance(endpoint, str) else None
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Invalid model endpoint")
        if not isinstance(model, str) or not model.strip() or len(model) > 512:
            raise ValueError("Explicit model identity is required")
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or not 1 <= timeout <= 600
        ):
            raise ValueError("Invalid model request timeout")
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or not api_key_env.isidentifier()
        ):
            raise ValueError("Credentials must name an environment variable")
        self.endpoint, self.model = endpoint.rstrip("/"), model
        self.api_key_env, self.timeout = api_key_env, timeout

    def _request(self, route, payload, label):
        payload = json.dumps(
            dict(model=self.model, **payload), allow_nan=False
        ).encode()
        if len(payload) > 256000:
            raise ValueError("Model request exceeds 256KB")
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ValueError("Model API credential is missing")
            headers["Authorization"] = "Bearer " + key
        try:
            request = Request(self.endpoint + route, data=payload, headers=headers)
            with build_opener(_NoRedirect()).open(
                request, timeout=self.timeout
            ) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("Oversized model response")
            return json.loads(raw)
        except Exception:
            raise ValueError(f"{label} server request failed") from None


def _signature(model, revision, **settings):
    # Servers may identify immutable versions by date/tag, not just Git hashes.
    # Identity is declared by the operator; this client cannot attest weights.
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or len(revision) > 512
        or revision.lower() in {"main", "master", "latest"}
    ):
        raise ValueError("Explicit immutable model revision is required")
    return json.dumps(dict(model=model, revision=revision, **settings), sort_keys=True)


def _template(value):
    if not isinstance(value, str) or len(value) > 4000:
        raise ValueError("Invalid embedding template")
    fields = list(Formatter().parse(value))
    if sum(field == "text" for _, field, _, _ in fields) != 1 or any(
        field not in {None, "text"} or spec or conversion
        for _, field, spec, conversion in fields
    ):
        raise ValueError("Embedding template requires exactly one plain {text}")
    return value


class EmbeddingServer(_Server):
    def __init__(
        self,
        endpoint,
        *,
        model,
        revision,
        query_template="{text}",
        passage_template="{text}",
        instruction=None,
        batch_size=8,
        **kwargs,
    ):
        super().__init__(endpoint, model=model, **kwargs)
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 64
        ):
            raise ValueError("Invalid embedding batch size")
        if instruction is not None:
            if (
                not isinstance(instruction, str)
                or len(instruction) > 2000
                or query_template != "{text}"
            ):
                raise ValueError("Use an instruction or a query template, not both")
            query_template = (
                "Instruct: "
                + instruction.replace("{", "{{").replace("}", "}}")
                + "\nQuery: {text}"
                if instruction
                else "{text}"
            )
        self.query_template, self.passage_template = (
            _template(query_template),
            _template(passage_template),
        )
        self.batch_size = batch_size
        self.signature = _signature(
            model,
            revision,
            query_template=query_template,
            passage_template=passage_template,
            protocol="openai-embeddings:templates:v2",
        )

    def encode_queries(self, texts):
        return self._encode(texts, self.query_template)

    def encode(self, texts):
        return self._encode(texts, self.passage_template)

    def _encode(self, texts, template):
        if any(not isinstance(t, str) for t in texts):
            raise ValueError("Embedding inputs must be strings")
        vectors = []
        for offset in range(0, len(texts), self.batch_size):
            batch = [
                template.format(text=t)
                for t in texts[offset : offset + self.batch_size]
            ]
            try:
                data = self._request("/embeddings", dict(input=batch), "Embedding")[
                    "data"
                ]
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
                raise ValueError("Embedding server request failed") from None
        if vectors and any(len(v) != len(vectors[0]) for v in vectors):
            raise ValueError("Embedding dimensions changed within request")
        return vectors


class RerankingServer(_Server):
    """Indexed relevance_score protocol used by compatible /rerank servers."""

    def __init__(self, endpoint, *, model, revision, **kwargs):
        super().__init__(endpoint, model=model, **kwargs)
        self.signature = _signature(model, revision, protocol="indexed-rerank:v1")

    def score(self, question, texts):
        if not isinstance(question, str) or any(not isinstance(t, str) for t in texts):
            raise ValueError("Reranking inputs must be strings")
        if not texts:
            return []
        try:
            results = self._request(
                "/rerank",
                dict(query=question, documents=list(texts), top_n=len(texts)),
                "Reranking",
            )["results"]
            if not isinstance(results, list) or len(results) != len(texts):
                raise ValueError("Incomplete reranking response")
            scores = [None] * len(texts)
            for row in results:
                i, score = row["index"], row["relevance_score"]
                if (
                    not isinstance(i, int)
                    or isinstance(i, bool)
                    or not 0 <= i < len(texts)
                    or scores[i] is not None
                    or not isinstance(score, (float, int))
                    or isinstance(score, bool)
                    or not math.isfinite(score)
                ):
                    raise ValueError("Invalid reranking response")
                scores[i] = score
            return scores
        except Exception:
            raise ValueError("Reranking server request failed") from None


class ChatServer(_Server):
    def __init__(
        self,
        endpoint,
        *,
        model,
        max_tokens=1500,
        temperature=0,
        json_mode=True,
        **kwargs,
    ):
        super().__init__(endpoint, model=model, **kwargs)
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= 100000
            or not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
            or not isinstance(json_mode, bool)
        ):
            raise ValueError("Invalid generation settings")
        self.max_tokens, self.temperature, self.json_mode = (
            max_tokens,
            temperature,
            json_mode,
        )

    def __call__(self, system, evidence):
        payload = dict(
            messages=[
                dict(role="system", content=system),
                dict(role="user", content=evidence),
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            content = self._request("/chat/completions", payload, "Generation")[
                "choices"
            ][0]["message"]["content"]
            if not isinstance(content, str) or len(content) > 50000:
                raise ValueError("Invalid generation response")
            return content
        except Exception:
            raise ValueError("Generation server request failed") from None
