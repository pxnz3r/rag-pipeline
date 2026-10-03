import json

import pytest

from rag_pipeline import Index, Navigation


def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    for name, company in [("a", "A"), ("b", "B")]:
        (root / f"{name}.csv").write_text("topic,value\nrevenue,100\nmargin,20\n")
        (root / f"{name}.csv.meta.json").write_text('{"company":"' + company + '"}')
    index = Index(tmp_path / "index.sqlite")
    index.ingest(root)
    return index, root


def test_navigation_scope_exact_reads_pagination_and_trace(tmp_path):
    index, _ = corpus(tmp_path)
    with index:
        session = Navigation(index, filters={"company": "A"})
        docs = session.documents(limit=1)
        assert [d["id"] for d in docs["documents"]] == ["a.csv"]
        assert docs["next_offset"] is None
        first = session.outline("a.csv", limit=1)
        second = session.outline("a.csv", offset=first["next_offset"], limit=1)
        assert first["sections"][0]["locator"] != second["sections"][0]["locator"]
        locator = first["sections"][0]["locator"]
        found = session.read("a.csv", locator, start=2, chars=5)
        source = found["source"]
        raw = index.db.execute(
            "SELECT text FROM sections WHERE document=? AND locator=?",
            ("a.csv", locator),
        ).fetchone()[0]
        assert source["text"] == raw[2:7] and source["end"] == 7
        assert session.read("b.csv", locator) == {"status": "not_found"}
        assert session.outline("b.csv")["sections"] == []
        assert {h["document"] for h in session.search("revenue", mode="lexical")} == {
            "a.csv"
        }
        with pytest.raises(ValueError, match="overridden"):
            session.search("revenue", filters={})
        assert [t["action"] for t in session.trace][-1] == "search"


def test_navigation_budgets_and_concurrent_changes(tmp_path):
    index, root = corpus(tmp_path)
    with index:
        session = Navigation(index, max_calls=1)
        session.documents()
        with pytest.raises(ValueError, match="call budget"):
            session.documents()
        session = Navigation(index, max_chars=100)
        with pytest.raises(ValueError, match="character budget"):
            session.documents()
        assert session.remaining == 100 and session.trace == []
        session = Navigation(index)
        with Index(index.path) as writer:
            (root / "a.csv").write_text("topic,value\nrevenue,101\n")
            writer.ingest(root)
        with pytest.raises(ValueError, match="changed"):
            session.documents()
        fresh = Navigation(index)
        assert len(fresh.documents()["documents"]) == 2


def test_agent_reads_original_evidence_and_reuses_citation_validation(tmp_path):
    index, _ = corpus(tmp_path)
    with index:
        session = Navigation(index, filters={"company": "A"})
        actions = iter(
            [
                {"action": "outline", "arguments": {"document": "a.csv"}},
                {
                    "action": "read",
                    "arguments": {"document": "a.csv", "locator": "row 2"},
                },
                {"action": "answer", "arguments": {}},
            ]
        )

        def plan(system, payload):
            assert "untrusted" in system
            assert json.loads(payload)["question"] == "Revenue?"
            return json.dumps(next(actions))

        def generate(system, payload):
            evidence = json.loads(payload)["evidence"]
            assert len(evidence) == 1 and evidence[0]["document"] == "a.csv"
            return json.dumps(
                {
                    "claims": [
                        {
                            "text": "Revenue was 999.",
                            "evidence": [
                                {
                                    "source_id": evidence[0]["id"],
                                    "quote": evidence[0]["text"],
                                }
                            ],
                        }
                    ]
                }
            )

        result = session.answer("Revenue?", plan=plan, generate=generate)
        assert result.status == "generation_failed"
        assert result.sources[0].text.endswith("value: 100")
        assert [t["action"] for t in session.trace] == ["outline", "read"]


def test_agent_cannot_expand_scope_or_loop_indefinitely(tmp_path):
    index, _ = corpus(tmp_path)
    with index:
        session = Navigation(index, filters={"company": "A"}, max_calls=2)
        actions = iter(
            [
                {
                    "action": "read",
                    "arguments": {"document": "b.csv", "locator": "row 2"},
                },
                {"action": "answer", "arguments": {}},
            ]
        )
        result = session.answer(
            "Revenue?", plan=lambda *args: json.dumps(next(actions))
        )
        assert result.status == "no_evidence"
        calls = []

        def repeated(*args):
            calls.append(1)
            return json.dumps({"action": "documents", "arguments": {}})

        with pytest.raises(ValueError, match="call budget"):
            Navigation(index, max_calls=2).answer("Revenue?", plan=repeated)
        assert len(calls) == 3


def test_agent_arithmetic_is_source_bound_and_returns_computation_provenance(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "report.txt").write_text("Revenue USD millions: 2024=125.0; 2023=100.0.")
    with Index(tmp_path / "i.sqlite") as index:
        index.ingest(root)
        session = Navigation(index)
        source = session.read("report.txt", "text")["source"]
        operands = [
            {
                "source_id": source["id"],
                "quote": source["text"],
                "value": v,
                "unit": "USD millions",
            }
            for v in ["125.0", "100.0"]
        ]
        actions = iter(
            [
                {
                    "action": "calculate",
                    "arguments": {"operation": "growth_percent", "operands": operands},
                },
                {"action": "answer", "arguments": {}},
            ]
        )
        result = session.answer(
            "Revenue growth?", plan=lambda *args: json.dumps(next(actions))
        )
        assert result.status == "calculated"
        calculation = json.loads(result.answer)["calculations"][0]
        assert calculation["value"] == "25.00" and calculation["unit"] == "%"
        assert calculation["operands"] == operands
        assert result.sources[0].text == source["text"]
        forged = [dict(operands[0], source_id="missing"), operands[1]]
        with pytest.raises(ValueError, match="exact source quote"):
            session.calculate("growth_percent", forged)


def test_heading_tree_original_ranges_scope_and_cascade(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    text = "# C#\nOverview.\n## Disposal\nCall Dispose.\n```\n# Fake\n```\n# C++\nUse RAII.\n"
    (root / "manual.md").write_text(text)
    (root / "manual.md.meta.json").write_text('{"company":"A"}')
    with Index(tmp_path / "i.sqlite") as index:
        index.ingest(root, contextual=True)
        session = Navigation(index, filters={"company": "A"})
        roots = session.headings("manual.md")["nodes"]
        assert [n["title"] for n in roots] == ["C#", "C++"]
        child = session.headings("manual.md", parent=roots[0]["id"])["nodes"][0]
        assert child["title"] == "Disposal"
        assert text[child["start"] : child["end"]].startswith("## Disposal")
        assert child["end"] == roots[0]["end"] == text.index("# C++")
        assert (
            Navigation(index, filters={"company": "B"}).headings("manual.md")["nodes"]
            == []
        )
        assert (
            Navigation(index, filters={"company": "B"}).headings(
                "manual.md", parent=roots[0]["id"]
            )["nodes"]
            == []
        )
        assert (
            session.read("manual.md", child["locator"], start=child["start"], chars=15)[
                "source"
            ]["text"]
            == text[child["start"] : child["start"] + 15]
        )
        (root / "manual.md").unlink()
        index.ingest(root, contextual=True)
        assert index.db.execute("SELECT COUNT(*) FROM heading_nodes").fetchone()[0] == 0


@pytest.mark.parametrize(
    "options", [{"max_calls": True}, {"max_chars": 0}, {"max_calls": 101}]
)
def test_invalid_navigation_budget(tmp_path, options):
    with Index(tmp_path / "i.sqlite") as index:
        with pytest.raises(ValueError):
            Navigation(index, **options)
