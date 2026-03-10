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
            def create(messages, model):
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
