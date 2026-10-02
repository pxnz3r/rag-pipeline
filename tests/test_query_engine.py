import asyncio

from rag_pipeline.query_engine import (
    generate_trading_answer_robust,
    run_evaluation_pipeline,
)


class FakeCollection:
    def query(self, query_texts, n_results):
        return {"ids": [["a", "b"]]}

    def get(self, ids, include):
        return {"documents": ["doc a", "doc b"]}


class FakeReranker:
    def predict(self, pairs, batch_size=16):
        # Prefer second doc
        return [0.1, 0.9]


class FakeClient:
    class Chat:
        class Completions:
            @staticmethod
            def create(messages, model, **kwargs):
                class Msg:
                    content = "answer text"

                class Choice:
                    message = Msg()

                class Resp:
                    choices = [Choice()]

                return Resp()

        completions = Completions()

    chat = Chat()


def test_generate_trading_answer_robust_basic():
    async def _run():
        ans, ctx = await generate_trading_answer_robust(
            "query",
            FakeCollection(),
            rag_instance=None,
            bm25_index=None,
            bm25_id_map={},
            retrieval_top_k=5,
            rrf_k=60,
            rerank_candidates=2,
            rerank_batch_size=8,
            final_top_k=1,
            context_max_chars=100,
            tokenize_for_bm25_fn=lambda q: q.split(),
            get_eval_components_fn=lambda: (None, None, FakeReranker()),
            get_cached_groq_client_fn=lambda: FakeClient(),
            query_param_factory=None,
            logger=None,
        )
        assert ans == "answer text"
        assert ctx == ["doc b"]

    asyncio.run(_run())


def test_run_evaluation_pipeline_basic():
    async def _run():
        async def generate_answer_fn(question, collection, rag):
            return f"ans:{question}", [f"ctx:{question}"]

        def dataset_from_dict_fn(data):
            return data

        def evaluate_fn(dataset, metrics, llm, embeddings):
            return {"rows": len(dataset["question"]), "metric_count": len(metrics)}

        result = await run_evaluation_pipeline(
            ["q1", "q2"],
            ["g1", "g2"],
            collection=None,
            rag_instance=None,
            generate_answer_fn=generate_answer_fn,
            dataset_from_dict_fn=dataset_from_dict_fn,
            evaluate_fn=evaluate_fn,
            metrics=["m1", "m2"],
            get_eval_components_fn=lambda: ("llm", "emb", None),
            throttle_sec=0.0,
        )
        assert result == {"rows": 2, "metric_count": 2}

    asyncio.run(_run())


def test_dense_failure_uses_bm25_and_citations_keep_document_alignment():
    from rag_pipeline.query_engine import query_answer

    class DenseDown(FakeCollection):
        def query(self, **kwargs):
            raise RuntimeError("vector service down")

        def get(self, ids, include):
            return {
                "ids": ["b", "a"],
                "documents": ["doc b", "doc a"],
                "metadatas": [
                    {"source": "B.pdf", "page": 2},
                    {"source": "A.pdf", "page": 1},
                ],
            }

    class BM25:
        def get_scores(self, tokens):
            return [1.0, 2.0]

    async def run():
        result = await query_answer(
            "query",
            DenseDown(),
            bm25_index=BM25(),
            bm25_id_map={0: "a", 1: "b"},
            tokenize_for_bm25_fn=lambda q: [q],
            final_top_k=2,
            get_cached_groq_client_fn=lambda: FakeClient(),
        )
        assert result.answer == "answer text"
        assert result.contexts == ["doc b", "doc a"]
        assert [s.pdf_name for s in result.sources] == ["B.pdf", "A.pdf"]
        assert [s.page_number for s in result.sources] == [2, 1]

    asyncio.run(run())


def test_no_context_does_not_call_generation():
    from rag_pipeline.query_engine import query_answer

    class Empty:
        def count(self):
            return 0

    def must_not_call():
        raise AssertionError("Generation should not run without evidence")

    result = asyncio.run(
        query_answer("query", Empty(), get_cached_groq_client_fn=must_not_call)
    )
    assert result.error == "no_context"


def test_generation_errors_do_not_disclose_vendor_error():
    from types import SimpleNamespace

    from rag_pipeline.query_engine import query_answer

    def fail(**kwargs):
        raise RuntimeError("secret-key-and-private-document-content")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )
    result = asyncio.run(
        query_answer(
            "query", FakeCollection(), get_cached_groq_client_fn=lambda: client
        )
    )
    assert result.error == "generation_failed"
    assert "secret-key" not in result.answer
    assert result.contexts


def test_retrieval_and_generation_do_not_block_event_loop():
    import time

    from rag_pipeline.query_engine import query_answer

    class Slow(FakeCollection):
        def query(self, **kwargs):
            time.sleep(0.05)
            return super().query(**kwargs)

    async def run():
        ticks = []

        async def heartbeat():
            for _ in range(5):
                await asyncio.sleep(0.005)
                ticks.append(True)

        task = asyncio.create_task(
            query_answer(
                "query", Slow(), get_cached_groq_client_fn=lambda: FakeClient()
            )
        )
        await heartbeat()
        assert not task.done()
        result = await task
        assert result.error is None
        assert len(ticks) == 5

    asyncio.run(run())


def test_context_bound_applies_to_reranking_prompt_and_returned_evidence():
    import json
    from types import SimpleNamespace

    from rag_pipeline.query_engine import query_answer

    captured = {}

    class Reranker:
        def predict(self, pairs, batch_size):
            captured["pairs"] = pairs
            return [0.1, 0.9]

    def generate(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=generate))
    )
    result = asyncio.run(
        query_answer(
            "query",
            FakeCollection(),
            context_max_chars=3,
            get_reranker_fn=Reranker,
            get_cached_groq_client_fn=lambda: client,
        )
    )
    assert all(len(pair[1]) <= 3 for pair in captured["pairs"])
    assert all(len(c) <= 3 for c in result.contexts)
    assert captured["messages"][0]["role"] == "system"
    evidence = json.loads(captured["messages"][1]["content"])
    assert evidence["documents"][0]["text"] == result.contexts[0]


def test_evaluation_rejects_mismatch_before_api_calls():
    import pytest

    async def run():
        with pytest.raises(ValueError, match="equal lengths"):
            await run_evaluation_pipeline(
                ["question"],
                [],
                None,
                None,
                generate_answer_fn=None,
                dataset_from_dict_fn=None,
                evaluate_fn=None,
                metrics=[],
                get_eval_components_fn=None,
            )

    asyncio.run(run())
