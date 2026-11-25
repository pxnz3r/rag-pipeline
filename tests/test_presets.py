import json
import os
import subprocess
import sys

import pytest

from rag_pipeline.configuration import load_config
from rag_pipeline.presets import PRESETS, config, preset


def run(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "rag_pipeline.cli", *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_presets_are_valid_configurations(tmp_path):
    for name in PRESETS:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(config(name)))
        assert load_config(path) == config(name)
    # Callers get a copy, never the shared definition.
    config("hybrid")["search"]["mode"] = "dense"
    assert PRESETS["hybrid"]["search"]["mode"] == "hybrid"
    with pytest.raises(ValueError, match="Unknown preset"):
        config("best")


def test_lexical_preset_and_config_override_through_cli(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text("Revenue was USD 125.0 million in 2024.")
    (root / "b.txt").write_text("Revenue fell. Lease costs rose.")
    path = tmp_path / "index.sqlite"
    ingest = run("--preset", "lexical", "--index", path, "ingest", root, cwd=tmp_path)
    assert ingest.returncode == 0
    proc = run(
        "--preset", "lexical", "--index", path, "search", "revenue", cwd=tmp_path
    )
    assert proc.returncode == 0 and len(json.loads(proc.stdout)) == 2
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"search": {"k": 1}}))
    proc = run(
        "--preset",
        "lexical",
        "--config",
        override,
        "--index",
        path,
        "search",
        "revenue",
        cwd=tmp_path,
    )
    assert proc.returncode == 0 and len(json.loads(proc.stdout)) == 1
    assert run("--preset", "best", "status", cwd=tmp_path).returncode == 2


@pytest.mark.skipif(
    os.environ.get("RAG_TEST_MODELS") != "1", reason="Opt-in public model download"
)
def test_hybrid_preset_finds_paraphrases(tmp_path):
    from rag_pipeline import Index

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "sales.txt").write_text("Quarterly turnover grew by eleven percent.")
    (root / "lease.txt").write_text("The tenant may terminate the lease early.")
    stack = preset("hybrid")
    with Index(tmp_path / "index.sqlite", stack.embedder, stack.reranker) as index:
        index.ingest(root)
        hits = index.search("How much did revenue increase?", **stack.search)
    assert hits[0].document == "sales.txt"
