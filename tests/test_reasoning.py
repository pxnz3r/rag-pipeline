import json
from decimal import Decimal

import pytest

from rag_pipeline import Hit, execute_program, reason_from_sources


@pytest.fixture
def evidence():
    text = "Revenue USD millions: 2024=125.0; 2023=100.0. Rate was 5% and term was 2 years."
    return [Hit("report#row", "report", "row", 0, len(text), text, {}, 0)]


def leaf(evidence, value):
    return dict(source_id=evidence[0].id, quote=evidence[0].text, value=value)


def test_program_growth_uses_grounded_operands_and_step_references(evidence):
    program = dict(
        steps=[
            dict(
                op="subtract", args=[leaf(evidence, "125.0"), leaf(evidence, "100.0")]
            ),
            dict(op="divide", args=[dict(step=0), leaf(evidence, "100.0")]),
        ]
    )
    result = execute_program(program, evidence)
    assert result.status == "calculated" and Decimal(result.value) == Decimal(".25")
    assert result.sources == evidence and len(result.steps) == 2
    assert result.steps[1]["args"][0] == dict(step=0)
    for op, values, expected in [
        ("multiply", ["100.0", "125.0"], "12500"),
        ("mean", ["100.0", "125.0"], "112.5"),
        ("min", ["100.0", "125.0"], "100"),
        ("max", ["100.0", "125.0"], "125"),
        ("count", ["100.0", "125.0"], "2"),
    ]:
        assert Decimal(
            execute_program(
                dict(steps=[dict(op=op, args=[leaf(evidence, v) for v in values])]),
                evidence,
            ).value
        ) == Decimal(expected)


@pytest.mark.parametrize(
    "arg",
    [
        dict(step=0),
        dict(step=True),
        dict(constant="invented"),
        dict(source_id="forged", quote="125.0", value="125.0"),
        dict(source_id="report#row", quote="2024=999.0", value="999.0"),
        dict(
            source_id="report#row",
            quote="Rate was 5% and term was 2 years.",
            value="2%",
        ),
    ],
)
def test_program_rejects_circular_forged_or_unit_swapped_operands(evidence, arg):
    with pytest.raises(ValueError):
        execute_program(dict(steps=[dict(op="identity", args=[arg])]), evidence)


def test_program_decimal_precision_percent_and_resource_limits(evidence):
    assert Decimal(
        execute_program(
            dict(steps=[dict(op="identity", args=[leaf(evidence, "5%")])]), evidence
        ).value
    ) == Decimal(".05")
    with pytest.raises(ValueError, match="zero"):
        execute_program(
            dict(
                steps=[
                    dict(
                        op="divide",
                        args=[leaf(evidence, "100.0"), dict(constant="zero")],
                    )
                ]
            ),
            evidence,
        )
    with pytest.raises(ValueError, match="exponent"):
        execute_program(
            dict(
                steps=[
                    dict(
                        op="power",
                        args=[leaf(evidence, "100.0"), leaf(evidence, "2024")],
                    )
                ]
            ),
            evidence,
        )
    text = "Amounts 1e50 and 1e-50."
    hit = Hit("wide", "report", "row", 0, len(text), text, {}, 0)
    result = execute_program(
        dict(
            steps=[
                dict(
                    op="add",
                    args=[
                        dict(source_id="wide", quote=text, value="1e50"),
                        dict(source_id="wide", quote=text, value="1e-50"),
                    ],
                )
            ]
        ),
        [hit],
    )
    assert result.value.endswith("00000000000000000000000000000000000000000000000001")
    with pytest.raises(ValueError):
        execute_program(
            dict(steps=[dict(op="eval", args=[leaf(evidence, "100")])]), evidence
        )


def test_reasoning_retries_bounded_validation_and_abstains_without_evidence(evidence):
    calls = []

    def generate(system, payload):
        calls.append(json.loads(payload))
        return json.dumps(
            dict(
                steps=[
                    dict(
                        op="identity",
                        args=[leaf(evidence, "999" if len(calls) == 1 else "125.0")],
                    )
                ]
            )
        )

    result = reason_from_sources(
        "Revenue?", evidence, generate=generate, max_attempts=2
    )
    assert (
        result.status == "calculated"
        and result.attempts == 2
        and Decimal(result.value) == 125
    )
    assert "feedback" in calls[1] and "999" not in calls[1]["feedback"]
    assert (
        reason_from_sources("Missing?", [], generate=generate).status == "no_evidence"
    )
    assert len(calls) == 2
    assert (
        reason_from_sources(
            "Revenue?", evidence, generate=lambda *a: '{"steps":[]}'
        ).status
        == "abstained"
    )


def test_catalog_resolves_exact_spans_and_preserves_trace(evidence):
    from rag_pipeline.reasoning import numeric_catalog, program_evidence

    catalog = numeric_catalog(evidence)
    ids = {item["value"]: key for key, item in catalog.items()}
    result = execute_program(
        dict(steps=[dict(op="identity", args=[dict(operand=ids["5%"])])]), evidence
    )
    assert Decimal(result.value) == Decimal(".05")
    assert result.steps[0]["args"][0]["value"] == "5%"
    assert result.steps[0]["args"][0]["quote"] in evidence[0].text
    for key, item in catalog.items():
        assert f"[{key}]" in program_evidence(evidence)[0]["text"]
        assert (
            evidence[0].text[item["start"] : item["end"]].strip().replace(" ", "")
            == item["value"]
        )
    with pytest.raises(ValueError, match="Unknown"):
        execute_program(
            dict(steps=[dict(op="identity", args=[dict(operand="N999")])]), evidence
        )


def test_json_lexemes_and_duplicate_keys():
    from rag_pipeline.reasoning import parse_program

    assert parse_program('{"steps":[{"op":"identity","args":[{"step":0}]}]}')["steps"][
        0
    ]["args"][0] == dict(step=0)
    assert (
        parse_program('{"steps":[],"number":1.2300000000000000001}')["number"]
        == "1.2300000000000000001"
    )
    for raw in ['{"steps":[],"steps":[]}', '{"steps":[],"number":NaN}']:
        with pytest.raises(ValueError):
            parse_program(raw)


def test_context_route_respects_scope_budget_and_source_revision(tmp_path):
    from rag_pipeline import Index, reason
    from rag_pipeline.reasoning import ContextBudgetExceeded

    folder = tmp_path / "corpus"
    folder.mkdir()
    source = folder / "report.txt"
    source.write_text("Revenue 125.0; last year 100.0." + " Financial report." * 10)
    (folder / "report.txt.meta.json").write_text('{"report":"annual"}')
    with Index(tmp_path / "index.sqlite") as index:
        index.ingest(folder)
        hit = index.search("Revenue")[0]
        assert hit.source_revision

        def generator(*args):
            return '{"steps":[{"op":"identity","args":[{"operand":"N0"}]}]}'

        result = reason(
            index,
            "Revenue?",
            generate=generator,
            route="context",
            filters={"report": "annual"},
        )
        assert (
            result.status == "calculated"
            and result.sources[0].source_revision == hit.source_revision
        )
        with pytest.raises(ContextBudgetExceeded):
            reason(
                index,
                "Revenue?",
                generate=generator,
                route="context",
                filters={"report": "annual"},
                max_evidence_chars=100,
            )
        result = reason(
            index,
            "Revenue?",
            generate=generator,
            route="auto",
            filters={"report": "annual"},
            max_evidence_chars=100,
        )
        assert result.status in {"calculated", "program_failed"}
        source.write_text("Revenue 130.0; last year 100.0.")
        index.ingest(folder)
        assert index.search("Revenue")[0].source_revision != hit.source_revision


def test_numeric_grounding_rejects_substrings_and_clipped_windows():
    from rag_pipeline.answers import _pack_evidence
    from rag_pipeline.reasoning import numeric_catalog

    hit = Hit("s", "report", "text", 0, 11, "Amount 1000", {}, 0)
    with pytest.raises(ValueError, match="absent"):
        execute_program(
            dict(
                steps=[
                    dict(
                        op="identity",
                        args=[dict(source_id="s", quote="100", value="100")],
                    )
                ]
            ),
            [hit],
        )
    clipped = Hit("c", "report", "text", 7, 9, "10", {}, 0, source_suffix="00 dollars")
    assert numeric_catalog([clipped]) == {}
    clipped = Hit("c", "report", "text", 8, 11, "000", {}, 0, source_prefix="Amount 1")
    assert numeric_catalog([clipped]) == {}
    text = " " * 98 + "1000 dollars"
    packed = _pack_evidence(
        [Hit("s", "report", "text", 0, len(text), text, {}, 0)], 100
    )
    assert numeric_catalog(packed) == {}
    clipped = Hit("c", "report", "text", 0, 1, "5", {}, 0, source_suffix=" percent")
    assert numeric_catalog([clipped]) == {}


def test_final_result_requires_source_dependency(evidence):
    with pytest.raises(ValueError, match="Final result"):
        execute_program(
            dict(
                steps=[
                    dict(op="identity", args=[leaf(evidence, "125.0")]),
                    dict(op="add", args=[dict(constant="one"), dict(constant="one")]),
                ]
            ),
            evidence,
        )


def test_fractional_power_supports_compound_growth():
    text = "Value grew from 100 to 125 over 4 years."
    hit = Hit("growth", "report", "row", 0, len(text), text, {}, 0)

    def operand(value):
        return dict(source_id=hit.id, quote=text, value=value)

    result = execute_program(
        dict(
            steps=[
                dict(op="divide", args=[operand("125"), operand("100")]),
                dict(op="divide", args=[dict(constant="one"), operand("4")]),
                dict(op="power", args=[dict(step=0), dict(step=1)]),
                dict(op="subtract", args=[dict(step=2), dict(constant="one")]),
            ]
        ),
        [hit],
    )
    assert round(Decimal(result.value), 15) == Decimal("0.057371263440564")


def test_retrieval_and_navigation_keep_numeric_boundary_context(tmp_path):
    from rag_pipeline import Index, Navigation
    from rag_pipeline.reasoning import numeric_catalog

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "revenue " + "x " * 45 + "123456789 USD"
    (corpus / "report.txt").write_text(text)
    with Index(tmp_path / "index.sqlite") as index:
        index.ingest(corpus, size=100, overlap=0)
        hit = index.search("revenue", context_chars=100)[0]
        assert hit.text.endswith("12") and hit.source_suffix.startswith("3456789")
        assert numeric_catalog([hit]) == {}
        read = Navigation(index).read("report.txt", "text", start=98, chars=2)
        assert read["source"]["text"] == "12"
        assert numeric_catalog([Hit(**read["source"])]) == {}


def test_unicode_minus_thousands_and_scientific_notation():
    from rag_pipeline.reasoning import numeric_catalog

    text = "Balance − 1\u202f234.50 USD; scale 1e−3."
    hit = Hit("unicode", "report", "row", 0, len(text), text, {}, 0)
    catalog = numeric_catalog([hit])
    assert [entry["value"] for entry in catalog.values()] == ["− 1\u202f234.50", "1e−3"]
    assert Decimal(
        execute_program(
            dict(steps=[dict(op="identity", args=[dict(operand="N0")])]), [hit]
        ).value
    ) == Decimal("-1234.5")
    assert Decimal(
        execute_program(
            dict(steps=[dict(op="identity", args=[dict(operand="N1")])]), [hit]
        ).value
    ) == Decimal(".001")
    with pytest.raises(ValueError, match="absent"):
        execute_program(
            dict(
                steps=[
                    dict(
                        op="identity",
                        args=[
                            dict(
                                source_id="unicode",
                                quote="1\u202f234.50",
                                value="1234.50",
                            )
                        ],
                    )
                ]
            ),
            [hit],
        )


def test_decimal_doses_attached_units_and_identifier_boundaries():
    from rag_pipeline.reasoning import numeric_catalog

    text = "Dose .5mg then 12mcg; identifier A125 and 123abc; values 100, 200."
    hit = Hit("dose", "report", "row", 0, len(text), text, {}, 0)
    assert [entry["value"] for entry in numeric_catalog([hit]).values()] == [
        ".5",
        "12",
        "100",
        "200",
    ]


def test_deep_model_json_fails_as_validation_error():
    from rag_pipeline.reasoning import parse_program

    with pytest.raises(ValueError, match="nesting"):
        parse_program("[" * 2000 + "0" + "]" * 2000)


def test_explicit_large_context_budget_is_not_a_model_ceiling(tmp_path):
    from rag_pipeline import Index, Navigation, reason

    root = tmp_path / "source"
    root.mkdir()
    text = (
        "Revenue USD 125 million.\n" + "Original evidence. " * 4000 + "\nTAIL_EVIDENCE"
    )
    (root / "report.txt").write_text(text)
    captured = []
    with Index(tmp_path / "db.sqlite") as index:
        index.ingest(root)
        result = reason(
            index,
            "Revenue?",
            route="context",
            filters={"title": "report"},
            max_evidence_chars=len(text),
            generate=lambda system, payload: (
                captured.append(payload)
                or '{"steps":[{"op":"identity","args":[{"operand":"N0"}]}]}'
            ),
        )
        assert result.value == "125" and "TAIL_EVIDENCE" in captured[0]
        assert len(captured[0]) > 50000
        nav = Navigation(index, max_chars=200000, max_calls=101)
        read = nav.read("report.txt", "text", chars=len(text))
        assert read["source"]["text"] == text
        assert nav.remaining == 200000 - len(json.dumps(read, ensure_ascii=False))
        small = Navigation(index, max_chars=1000)
        with pytest.raises(ValueError, match="budget"):
            small.read("report.txt", "text", chars=10**100)
        assert small.remaining == 1000 and not small.trace
