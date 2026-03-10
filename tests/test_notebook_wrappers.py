import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace


def _cell_source(idx: int) -> str:
    nb = json.loads(Path("Python3finale.ipynb").read_text(encoding="utf-8"))
    return "".join(nb["cells"][idx]["source"])


def test_processing_cell_is_wrapper_based():
    src = _cell_source(5)
    assert (
        "from rag_pipeline.processing import process_pdf_page_level as process_pdf_page_level_pkg, sync_master_chunks"
        in src
    )
    assert "return process_pdf_page_level_pkg(" in src
    assert "all_chunks = sync_master_chunks(" in src


def test_enrichment_cell_is_wrapper_based():
    src = _cell_source(7)
    assert (
        "from rag_pipeline.enrichment import batch_enrich_chunks as batch_enrich_chunks_pkg"
        in src
    )
    assert "enriched = batch_enrich_chunks_pkg(" in src


def test_retrieval_and_chroma_cells_are_wrapper_based():
    src9 = _cell_source(9)
    src11 = _cell_source(11)
    assert (
        "from rag_pipeline.bm25_utils import build_bm25_index, tokenize_for_bm25"
        in src9
    )
    assert (
        "from rag_pipeline.chroma_pipeline import populate_chromadb as populate_chromadb_pkg"
        in src11
    )
    assert (
        "from rag_pipeline.chroma_pipeline import purge_stale_data as purge_stale_data_pkg"
        in src11
    )


def test_query_eval_cell_is_wrapper_based():
    src = _cell_source(13)
    assert "from rag_pipeline.query_engine import (" in src
    assert "return await generate_trading_answer_robust_pkg(" in src
    assert "results = await run_evaluation_pipeline_pkg(" in src


def test_query_eval_wrappers_delegate_with_mocked_services():
    src = _cell_source(13)
    match = re.search(
        r"async def generate_trading_answer_robust\(query: str, collection, rag_instance=None\):.*",
        src,
        flags=re.S,
    )
    assert match is not None
    wrapper_code = match.group(0)

    captured = {}

    async def fake_generate_pkg(*args, **kwargs):
        captured["generate_called"] = True
        captured["generate_kwargs"] = kwargs
        return "wrapped-answer", ["ctx1"]

    async def fake_run_pkg(*args, **kwargs):
        captured["run_called"] = True
        captured["run_kwargs"] = kwargs
        return {"ok": True}

    ns = {
        "generate_trading_answer_robust_pkg": fake_generate_pkg,
        "run_evaluation_pipeline_pkg": fake_run_pkg,
        "bm25_index": None,
        "bm25_id_map": {},
        "RETRIEVAL_TOP_K": 50,
        "RRF_K": 60,
        "RERANK_CANDIDATES": 20,
        "RERANK_BATCH_SIZE": 16,
        "FINAL_TOP_K": 5,
        "CONTEXT_MAX_CHARS": 2000,
        "tokenize_for_bm25": lambda q: q.split(),
        "get_eval_components": lambda: ("llm", "emb", "rerank"),
        "get_cached_groq_client": lambda: object(),
        "QueryParam": lambda mode: SimpleNamespace(mode=mode),
        "logger": None,
        "Dataset": SimpleNamespace(from_dict=lambda x: x),
        "evaluate": lambda **kwargs: {"eval": True},
        "faithfulness": "m1",
        "answer_relevancy": "m2",
        "context_precision": "m3",
        "context_recall": "m4",
        "tqdm": lambda items: items,
        "EVAL_THROTTLE_SEC": 0.0,
    }
    exec(wrapper_code, ns)  # noqa: S102 - controlled test namespace

    async def _run():
        ans, ctx = await ns["generate_trading_answer_robust"](
            "q", collection=object(), rag_instance=None
        )
        assert ans == "wrapped-answer"
        assert ctx == ["ctx1"]

        res = await ns["run_evaluation_pipeline"](
            ["q1"], ["g1"], collection=object(), rag_instance=None
        )
        assert res == {"ok": True}

    asyncio.run(_run())
    assert captured.get("generate_called") is True
    assert captured.get("run_called") is True
