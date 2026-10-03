"""Optional pinned CPU models without PyTorch, pickle, or executable model code."""

from __future__ import annotations

import numpy as np


class _ONNX:
    batch_size = 8
    weights = "onnx/model.onnx"

    def __init__(self, threads=2, providers=None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        if (
            not isinstance(threads, int)
            or isinstance(threads, bool)
            or not 1 <= threads <= 64
        ):
            raise ValueError("Threads must be between 1 and 64")
        if providers is not None and (
            not isinstance(providers, list)
            or not providers
            or any(
                not isinstance(p, str) or p not in ort.get_available_providers()
                for p in providers
            )
        ):
            raise ValueError("Requested ONNX execution provider is unavailable")
        self.tokenizer = Tokenizer.from_file(
            hf_hub_download(self.model, "tokenizer.json", revision=self.revision)
        )
        self.tokenizer.enable_truncation(max_length=self.length)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            hf_hub_download(self.model, self.weights, revision=self.revision),
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )

    def _batches(self, texts):
        order = sorted(
            range(len(texts)),
            key=lambda i: (
                len(texts[i]) if isinstance(texts[i], str) else sum(map(len, texts[i]))
            ),
        )
        for offset in range(0, len(texts), self.batch_size):
            positions = order[offset : offset + self.batch_size]
            tokens = self.tokenizer.encode_batch([texts[i] for i in positions])
            feeds = {
                "input_ids": np.array([t.ids for t in tokens], dtype=np.int64),
                "attention_mask": np.array(
                    [t.attention_mask for t in tokens], dtype=np.int64
                ),
                "token_type_ids": np.array(
                    [t.type_ids for t in tokens], dtype=np.int64
                ),
            }
            yield (
                self.session.run(
                    [self.output_name] if hasattr(self, "output_name") else None,
                    {i.name: feeds[i.name] for i in self.session.get_inputs()},
                )[0],
                feeds["attention_mask"],
                positions,
            )


class MiniLM(_ONNX):
    """General English dense baseline, masked mean pooling, 256-token truncation."""

    model = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    length = 256
    signature = f"{model}@{revision}:onnx:mean:256:v1"

    def encode(self, texts):
        outputs = [None] * len(texts)
        for hidden, attention, positions in self._batches(texts):
            mask = attention[..., None]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clip(min=1)
            for position, vector in zip(positions, pooled):
                outputs[position] = vector
        return (
            np.asarray(outputs, dtype=np.float32)
            if outputs
            else np.empty((0, 384), dtype=np.float32)
        )


class CrossEncoder(_ONNX):
    """Optional MS MARCO reranker. Scores are logits, not confidence probabilities."""

    batch_size = 8

    model = "cross-encoder/ms-marco-MiniLM-L6-v2"
    revision = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    length = 512
    signature = f"{model}@{revision}:onnx:logits:512:v1"

    def score(self, question, texts):
        outputs = np.empty(len(texts), dtype=np.float32)
        for logits, _, positions in self._batches([(question, text) for text in texts]):
            outputs[positions] = logits.reshape(-1)
        return outputs


class E5(MiniLM):
    """English asymmetric retrieval encoder; query/passage prefixes are mandatory."""

    batch_size = 16
    model = "intfloat/e5-small-v2"
    revision = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
    weights = "onnx/model_qint8_avx512_vnni.onnx"
    length = 512
    signature = f"{model}@{revision}:onnx:qint8:mean:512:query-passage:v1"

    def encode(self, texts):
        return super().encode(["passage: " + text for text in texts])

    def encode_queries(self, texts):
        return super().encode(["query: " + text for text in texts])


class ColBERT(_ONNX):
    """Token MaxSim reranking, with the trained 96D projection in the graph.

    This is candidate reranking, not a PLAID index. Query mask augmentation and
    punctuation masking follow the checkpoint's ColBERT training configuration.
    """

    model = "answerdotai/answerai-colbert-small-v1"
    revision = "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e"
    # onnx/model.onnx exposes the backbone's 384D states, not ColBERT vectors.
    weights = "model.onnx"
    length = 300
    query_length = 32
    dimensions = 96
    query_prefix_id = 1
    document_prefix_id = 2
    pad_id = 0
    mask_id = 103
    query_expansion = True
    signature = f"{model}@{revision}:onnx:fp32:maxsim:32:300:96:v1"

    def _tokens(self, texts, query=False):
        import string

        length = self.query_length if query else self.length
        self.tokenizer.enable_truncation(max_length=length - 1)
        self.tokenizer.enable_padding(
            length=length if query and self.query_expansion else None,
            pad_id=self.mask_id if query and self.query_expansion else self.pad_id,
        )
        outputs = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        punctuation = {
            self.tokenizer.token_to_id(token) for token in string.punctuation
        }
        for offset in range(0, len(texts), self.batch_size):
            positions = order[offset : offset + self.batch_size]
            tokens = self.tokenizer.encode_batch([texts[i] for i in positions])
            ids = np.array([t.ids for t in tokens], dtype=np.int64)
            attention = np.array([t.attention_mask for t in tokens], dtype=np.int64)
            types = np.array([t.type_ids for t in tokens], dtype=np.int64)
            # Reserve exactly one position for [Q]/[D] after [CLS].
            ids = np.insert(
                ids,
                1,
                self.query_prefix_id if query else self.document_prefix_id,
                axis=1,
            )[:, :length]
            attention = np.insert(attention, 1, 1, axis=1)[:, :length]
            types = np.insert(types, 1, 0, axis=1)[:, :length]
            feeds = dict(input_ids=ids, attention_mask=attention, token_type_ids=types)
            hidden = self.session.run(
                None, {i.name: feeds[i.name] for i in self.session.get_inputs()}
            )[0]
            if (
                hidden.shape != (*ids.shape, self.dimensions)
                or not np.isfinite(hidden).all()
            ):
                raise ValueError("Invalid ColBERT token projection")
            for j, position in enumerate(positions):
                mask = (
                    np.ones(ids.shape[1], dtype=bool)
                    if query and self.query_expansion
                    else attention[j] > 0
                )
                if not query:
                    mask &= ~np.isin(ids[j], list(punctuation))
                vectors = hidden[j][mask]
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if not len(vectors) or (norms == 0).any():
                    raise ValueError("Invalid ColBERT token vectors")
                outputs[position] = vectors / norms
        return outputs

    def score(self, question, texts):
        if not texts:
            return np.empty(0, dtype=np.float32)
        return self.score_vectors(
            self.encode_query(question), self.encode_documents(texts)
        )

    def encode_query(self, question):
        return self._tokens([question], query=True)[0]

    def encode_documents(self, texts):
        return self._tokens(texts)

    @staticmethod
    def score_vectors(query, documents):
        return np.array(
            [np.max(query @ doc.T, axis=1).sum() for doc in documents],
            dtype=np.float32,
        )


class ModernColBERT(ColBERT):
    """ModernBERT late-interaction comparator; 48-token queries, no augmentation."""

    model = "lightonai/GTE-ModernColBERT-v1"
    revision = "25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639"
    query_length = 48
    dimensions = 128
    query_prefix_id = 50368
    document_prefix_id = 50369
    pad_id = 50283
    mask_id = 50284
    query_expansion = False
    signature = f"{model}@{revision}:onnx:fp32:maxsim:48:300:128:v1"


class ONNXEmbedding(_ONNX):
    """Configured ONNX graph, tokenizer, pooling and asymmetric text prefixes.

    Model-specific token protocols require their own installed adapter rather
    than guessing settings from the repository name.
    """

    def __init__(
        self,
        *,
        model,
        revision,
        dimensions,
        length,
        pooling,
        weights="onnx/model.onnx",
        query_prefix="",
        passage_prefix="",
        output_index=0,
        batch_size=8,
        threads=2,
        providers=None,
    ):
        import json
        import re

        if (
            not isinstance(model, str)
            or not model.strip()
            or not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", revision)
        ):
            raise ValueError("ONNX model and immutable repository commit required")
        if (
            any(
                not isinstance(v, int) or isinstance(v, bool) or v < 1
                for v in (dimensions, length, batch_size)
            )
            or dimensions > 65536
            or length > 131072
            or batch_size > 256
        ):
            raise ValueError("Invalid ONNX dimensions, length or batch size")
        if (
            pooling not in {"mean", "cls", "last", "pooled"}
            or not isinstance(output_index, int)
            or isinstance(output_index, bool)
            or output_index < 0
        ):
            raise ValueError("Explicit supported pooling and output index required")
        if any(
            not isinstance(v, str) or len(v) > 4000
            for v in (query_prefix, passage_prefix, weights)
        ):
            raise ValueError("Invalid ONNX prefix or weight path")
        self.model, self.revision, self.weights = model, revision, weights
        self.dimensions, self.length, self.batch_size = dimensions, length, batch_size
        self.pooling, self.output_index = pooling, output_index
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.signature = json.dumps(
            dict(
                model=model,
                revision=revision,
                weights=weights,
                dimensions=dimensions,
                length=length,
                pooling=pooling,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
                output_index=output_index,
                protocol="onnx-config:v1",
            ),
            sort_keys=True,
        )
        super().__init__(threads=threads, providers=providers)
        outputs = self.session.get_outputs()
        if output_index >= len(outputs):
            raise ValueError("ONNX output index does not exist")
        self.output_name = outputs[output_index].name

    def encode(self, texts):
        return self._encode([self.passage_prefix + t for t in texts])

    def encode_queries(self, texts):
        return self._encode([self.query_prefix + t for t in texts])

    def _encode(self, texts):
        values = np.empty((len(texts), self.dimensions), dtype=np.float32)
        for hidden, attention, positions in self._batches(texts):
            if self.pooling == "pooled":
                pooled = hidden
            elif hidden.ndim != 3 or hidden.shape[:2] != attention.shape:
                raise ValueError("ONNX token output shape does not match attention")
            elif self.pooling == "mean":
                mask = attention[..., None]
                pooled = (hidden * mask).sum(1) / mask.sum(1).clip(min=1)
            elif self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                last = np.where(attention != 0, np.arange(attention.shape[1]), -1).max(
                    1
                )
                if (last < 0).any():
                    raise ValueError("ONNX query has no unmasked tokens")
                pooled = hidden[np.arange(len(hidden)), last]
            if (
                pooled.shape != (len(positions), self.dimensions)
                or not np.isfinite(pooled).all()
            ):
                raise ValueError("ONNX embedding output does not match configuration")
            values[positions] = pooled
        return values


class ONNXReranker(ONNXEmbedding):
    """Configured cross-encoder output; multiclass scoring is explicit."""

    def __init__(self, *, score_index=None, **options):
        if score_index is not None and (
            not isinstance(score_index, int)
            or isinstance(score_index, bool)
            or score_index < 0
        ):
            raise ValueError("Invalid reranking score index")
        super().__init__(dimensions=1, pooling="pooled", **options)
        import json

        self.score_index = score_index
        self.signature = json.dumps(
            dict(
                encoder=json.loads(self.signature),
                score_index=score_index,
                protocol="onnx-rerank:v1",
            ),
            sort_keys=True,
        )

    def score(self, question, texts):
        values = np.empty(len(texts), dtype=np.float32)
        for logits, _, positions in self._batches([(question, text) for text in texts]):
            if self.score_index is None:
                if logits.size != len(positions):
                    raise ValueError(
                        "Multiclass reranker requires explicit score_index"
                    )
                scores = logits.reshape(-1)
            else:
                if logits.ndim != 2 or self.score_index >= logits.shape[1]:
                    raise ValueError("Reranker score_index outside graph output")
                scores = logits[:, self.score_index]
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite ONNX reranking scores")
            values[positions] = scores
        return values
