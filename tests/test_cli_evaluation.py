import json
import subprocess
import sys
from pathlib import Path

import nbformat

from rag_pipeline.evaluation import evaluate


def run(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "rag_pipeline.cli", *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_installed_cli_ingest_search_ask_and_errors(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "book.txt").write_text("Revenue was USD 125.0 million in 2024.")
    path = tmp_path / "index.sqlite"
    assert run("--index", path, "ingest", root, cwd=tmp_path).returncode == 0
    proc = run(
        "--index", path, "search", "Revenue", "--filter", "title=book", cwd=tmp_path
    )
    assert proc.returncode == 0 and json.loads(proc.stdout)[0]["locator"] == "text"
    proc = run("--index", path, "ask", "Revenue", cwd=tmp_path)
    assert json.loads(proc.stdout)["status"] == "evidence"
    assert run("--index", path, "--dense", "status", cwd=tmp_path).returncode == 0
    assert (
        run(
            "--index",
            path,
            "--dense",
            "search",
            "Revenue",
            "--mode",
            "lexical",
            cwd=tmp_path,
        ).returncode
        == 0
    )
    assert (
        run(
            "--index", path, "search", "Revenue", "--filter", "malformed", cwd=tmp_path
        ).returncode
        == 1
    )
    assert (
        run("--index", tmp_path / "missing.sqlite", "status", cwd=tmp_path).returncode
        == 1
    )


def test_judged_domain_fixture_and_notebook():
    root = Path(__file__).resolve().parents[1]
    result = evaluate(
        root / "benchmarks/domain-canary.json", k=3, repeats=1, split="heldout"
    )
    assert (
        result["retrieval"]["recall_at_k"] == 1
        and result["negative_abstention_rate"] == 1
    )
    notebook = nbformat.read(root / "Python3finale.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<notebook>", "exec")


def test_character_metrics_union_and_wrong_document():
    from rag_pipeline.evaluation import span_metrics

    gold = [{"document": "a", "start": 10, "end": 20}]
    predicted = [
        {"document": "a", "start": 0, "end": 15},
        {"document": "a", "start": 5, "end": 20},
        {"document": "b", "start": 0, "end": 10},
    ]
    result = span_metrics(predicted, gold)
    assert result["character_recall"] == 1
    assert result["character_precision"] == 1 / 3 and result["retrieved_chars"] == 30
    assert span_metrics([], gold)["character_recall"] == 0
