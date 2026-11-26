import json
import subprocess
import sys

import pytest

from rag_pipeline import components
from rag_pipeline.capabilities import Capabilities, probe
from rag_pipeline.components import Component, auto_profile, resolve
from rag_pipeline.configuration import load_config

GIB = 1024**3
LAPTOP = Capabilities(cpus=4, memory_bytes=8 * GIB, extras=("models",))
GPU = Capabilities(
    cpus=16,
    memory_bytes=64 * GIB,
    onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    extras=("models",),
)


def test_auto_profile_follows_measured_hardware():
    assert auto_profile(LAPTOP) == "laptop"
    assert auto_profile(Capabilities(cpus=8, memory_bytes=16 * GIB)) == "workstation"
    assert auto_profile(GPU) == "server"


def test_chain_takes_first_runnable_and_reports_skips():
    plan = resolve("laptop", LAPTOP)
    assert plan.chosen["encoder"].name == "arctic-m"
    assert plan.chosen["reranker"].name == "none"
    skipped = plan.summary()["stages"]["encoder"]["skipped"]
    assert skipped["voyage"] == "network disabled (RAG_ALLOW_NETWORK=0)"
    with pytest.raises(ValueError, match="missing is not installed"):
        resolve("workstation", LAPTOP, pin={"reranker": "missing"})
    # Without the models extra the floor is BM25, and the plan says why.
    bare = resolve("laptop", Capabilities(cpus=2, memory_bytes=2 * GIB))
    assert bare.chosen["encoder"].name == "none"
    assert (
        "rag-pipeline[models]"
        in bare.summary()["stages"]["encoder"]["skipped"]["arctic-m"]
    )
    assert bare.config()["search"] == {"mode": "lexical"}


def test_accelerator_reaches_onnx_providers_and_config_is_valid(tmp_path):
    plan = resolve("server", GPU)
    settings = plan.config()
    assert settings["embedding"]["options"]["providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert plan.chosen["reranker"].name == "gte-reranker"
    assert settings["search"]["rerank_candidates"] == 20
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(settings))
    assert load_config(path) == settings


def test_hosted_components_need_key_and_network():
    keyed = Capabilities(
        cpus=4,
        memory_bytes=8 * GIB,
        services=("voyage",),
        network=True,
        extras=("models",),
    )
    assert resolve("laptop", keyed).chosen["encoder"].name == "voyage"
    offline = Capabilities(
        cpus=4, memory_bytes=8 * GIB, services=(), network=False, extras=("models",)
    )
    plan = resolve("laptop", offline)
    assert plan.chosen["encoder"].name == "arctic-m"
    assert (
        "network disabled" in plan.summary()["stages"]["encoder"]["skipped"]["voyage"]
    )
    caps = probe(network=False, environ={"VOYAGE_API_KEY": "x"})
    assert caps.services == () and not caps.network
    assert probe(network=True, environ={"VOYAGE_API_KEY": "x"}).services == ("voyage",)


def test_pinned_encoder_never_falls_back():
    with pytest.raises(ValueError, match="No runnable encoder"):
        resolve(
            "laptop",
            Capabilities(cpus=2, memory_bytes=2 * GIB),
            pin={"encoder": "arctic-m"},
        )
    assert (
        resolve("laptop", LAPTOP, pin={"encoder": "bge-small"}).chosen["encoder"].name
        == "bge-small"
    )


def test_plugins_register_through_entry_points():
    chat = Component("llama-server", "chat", {"kind": "openai", "options": {}})

    class Point:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def load(self):
            return lambda: self.value

    def points(group):
        assert group == "rag_pipeline.components"
        return [Point("b", [chat]), Point("a", [])]

    assert components._plugins(points) == [chat]
    with pytest.raises(ValueError, match="Component objects"):
        components._plugins(lambda group: [Point("bad", ["not a component"])])


def test_duplicate_component_names_fail(monkeypatch):
    duplicate = Component("arctic-m", "encoder", None)
    monkeypatch.setattr(components, "_plugins", lambda _: [duplicate])
    with pytest.raises(ValueError, match="Duplicate"):
        resolve("laptop", LAPTOP)


def run(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "rag_pipeline.cli", *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={"RAG_ALLOW_NETWORK": "0", "PATH": "/usr/bin:/bin"},
    )


def test_doctor_and_profile_record_the_plan(tmp_path):
    proc = run("doctor", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["capabilities"]["network"] is False
    assert set(report["plan"]["stages"]) == {"encoder", "reranker", "chat"}
    assert (
        run(
            "--preset", "lexical", "--profile", "laptop", "status", cwd=tmp_path
        ).returncode
        == 1
    )


def test_ingest_records_the_plan_with_the_publication(tmp_path):
    from rag_pipeline import Index

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text("Revenue was USD 125.0 million.")
    plan = resolve("laptop", Capabilities(cpus=2, memory_bytes=2 * GIB)).summary()
    with Index(tmp_path / "index.sqlite") as index:
        index.ingest(root, plan=plan)
        assert index.status()["plan"] == plan
        index.ingest(root)  # A later ingest without a plan keeps the record.
        assert index.status()["plan"]["stages"]["encoder"]["use"] == "none"
