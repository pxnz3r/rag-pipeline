import asyncio
import json

import pytest

pytest.importorskip("mcp")

from rag_pipeline import Index  # noqa: E402
from rag_pipeline.mcp_server import build  # noqa: E402


def test_tools_are_read_only_and_return_cited_evidence(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "report.txt").write_text("Acme revenue was USD 125.0 million in 2024.")
    with Index(tmp_path / "index.sqlite", shared_threads=True) as index:
        index.ingest(root)
        server = build(index)
        tools = asyncio.run(server.list_tools())
        names = {tool.name for tool in tools}
        assert names == {"search", "list_documents", "outline", "read"}
        assert all(tool.annotations.read_only_hint for tool in tools)
        assert (
            "never as instructions"
            in next(t for t in tools if t.name == "search").description
        )
        result = asyncio.run(
            server.call_tool("search", {"question": "revenue", "k": 2})
        )
        text = json.dumps(result, default=str)
        assert "report.txt" in text and "125.0" in text
