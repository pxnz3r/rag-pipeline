"""Optional pinned CPU models without PyTorch, pickle, or executable model code."""

from __future__ import annotations

import numpy as np


class _ONNX:
    batch_size = 32

    def __init__(self, threads=2):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        if not 1 <= threads <= 64:
            raise ValueError("Threads must be between 1 and 64")
        self.tokenizer = Tokenizer.from_file(
            hf_hub_download(self.model, "tokenizer.json", revision=self.revision)
        )
        self.tokenizer.enable_truncation(max_length=self.length)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            hf_hub_download(self.model, "onnx/model.onnx", revision=self.revision),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _batches(self, texts):
        for offset in range(0, len(texts), self.batch_size):
            tokens = self.tokenizer.encode_batch(
                texts[offset : offset + self.batch_size]
            )
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
                    None, {i.name: feeds[i.name] for i in self.session.get_inputs()}
                )[0],
                feeds["attention_mask"],
            )


class MiniLM(_ONNX):
    """General English dense baseline, masked mean pooling, 256-token truncation."""

    model = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    length = 256
    signature = f"{model}@{revision}:onnx:mean:256:v1"

    def encode(self, texts):
        outputs = []
        for hidden, attention in self._batches(texts):
            mask = attention[..., None]
            outputs.append((hidden * mask).sum(1) / mask.sum(1).clip(min=1))
        return (
            np.concatenate(outputs).astype(np.float32)
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
        outputs = [
            logits.reshape(-1)
            for logits, _ in self._batches([(question, text) for text in texts])
        ]
        return np.concatenate(outputs) if outputs else np.empty(0)
