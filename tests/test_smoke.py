from rag_pipeline.smoke import run_smoke


def test_offline_smoke_passes():
    result = run_smoke(live=False)
    assert result.ok is True
    assert result.mode == "offline"


def test_live_smoke_requires_env(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = run_smoke(live=True)
    assert result.ok is False
    assert "GROQ_API_KEY" in result.message
