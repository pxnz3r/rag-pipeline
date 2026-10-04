"""Real extraction/indexing with local dataset bytes: scope and holdout integrity."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


def test_report_disjoint_holdout_deduplicates_identical_views(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/prepare_qa.py"
    spec = importlib.util.spec_from_file_location("qa_preparation", path)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    data = {}
    for split in ("dev", "test"):
        records = [
            dict(
                id=f"{split}-{i}",
                filename=f"report-{i % 20}",
                pre_text=["Original evidence."],
                post_text=["Original subsequent text."],
                table=[["Account", "2024", "2023"], ["Revenue", "125", "100"]],
                qa=dict(
                    question="QUESTION_NOT_FOR_INDEXING",
                    exe_ans=987654321,
                    program="GOLD_NOT_FOR_INDEXING",
                ),
            )
            for i in range(100)
        ]
        data[split] = json.dumps(records).encode()
    monkeypatch.setattr(
        prepare, "HASHES", {k: hashlib.sha256(v).hexdigest() for k, v in data.items()}
    )
    monkeypatch.setattr(
        prepare.urllib.request,
        "urlopen",
        lambda url, **kw: io.BytesIO(data[Path(url).stem]),
    )
    monkeypatch.setattr(prepare, "load_config", lambda p: {"embedding": None})
    monkeypatch.setattr(prepare, "provider", lambda *a: None)
    exclusion = tmp_path / "excluded.json"
    exclusion.write_text(json.dumps(dict(queries=[], documents=[])))
    output = tmp_path / "qa"
    prepare.prepare(
        output, exclusion, None, exclude_test_reports=True, table_labels=True
    )
    selected = json.loads((output / "records.json").read_text())
    assert {r["filename"] for r in selected["dev"]}.isdisjoint(
        r["filename"] for r in selected["test"]
    )
    mapping = json.loads((output / "mapping.json").read_text())
    views = list((output / "corpus").glob("*.md"))
    assert len(views) == len(set(mapping.values())) < 32
    for file in views:
        text = file.read_text()
        assert "2024: 125" in text and "2023: 100" in text
        assert (
            "GOLD_NOT_FOR_INDEXING" not in text
            and "QUESTION_NOT_FOR_INDEXING" not in text
            and "987654321" not in text
        )
    with pytest.raises(ValueError, match="fresh"):
        prepare.prepare(output, exclusion, None)
