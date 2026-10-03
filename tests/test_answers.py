import json

import pytest

from rag_pipeline.answers import Operand, answer, calculate
from rag_pipeline.index import Hit

HIT = Hit(
    "report#row 2:0-70",
    "report",
    "row 2",
    0,
    70,
    "Revenue USD millions: 2024=125.0; 2023=100.0. Loss USD millions: (12.50).",
    {"company": "A"},
    1,
)


class Retriever:
    def search(self, *args, **kwargs):
        return [HIT]


def test_source_order_keeps_evidence_budget_and_numerical_row_order(tmp_path):
    from rag_pipeline import Index

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "report.csv").write_text(
        "topic,value\n" + "\n".join(f"revenue,{i}" for i in range(12))
    )
    with Index(tmp_path / "i.sqlite") as index:
        index.ingest(root)
        ranked = answer(index, "revenue", k=10, mode="lexical", max_evidence_chars=300)
        ordered = answer(
            index,
            "revenue",
            k=10,
            mode="lexical",
            max_evidence_chars=300,
            context_order="source",
        )
        assert {h.id for h in ranked.sources} == {h.id for h in ordered.sources}
        ordinals = [int(h.locator.split()[1]) for h in ordered.sources]
        assert ordinals == sorted(ordinals)
        assert sum(len(h.text) for h in ordered.sources) <= 300


def test_offline_evidence_and_valid_cited_response():
    result = answer(Retriever(), "Revenue?")
    assert result.status == "evidence" and HIT.text in result.answer
    quote = "Revenue USD millions: 2024=125.0; 2023=100.0."
    payload = {
        "claims": [
            {
                "text": "Revenue was 125.0 USD millions in 2024.",
                "evidence": [{"source_id": HIT.id, "quote": quote}],
            }
        ]
    }

    def generate(system, evidence):
        assert "untrusted" in system and json.loads(evidence)["question"] == "Revenue?"
        return json.dumps(payload)

    result = answer(Retriever(), "Revenue?", generate=generate)
    assert result.status == "cited" and result.claims == payload["claims"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "claims": [
                {
                    "text": "Revenue is 999.",
                    "evidence": [{"source_id": HIT.id, "quote": HIT.text}],
                }
            ]
        },
        {
            "claims": [
                {
                    "text": "Revenue is 125.0.",
                    "evidence": [{"source_id": "forged", "quote": HIT.text}],
                }
            ]
        },
        {
            "claims": [
                {
                    "text": "A claim",
                    "evidence": [{"source_id": HIT.id, "quote": "invented quote"}],
                }
            ]
        },
        {
            "claims": [
                {"text": "A claim", "evidence": [{"source_id": HIT.id, "quote": " "}]}
            ]
        },
        {
            "claims": [
                {
                    "text": "Revenue is 999m.",
                    "evidence": [{"source_id": HIT.id, "quote": HIT.text}],
                }
            ]
        },
        {
            "claims": [
                {
                    "text": "Revenue is 125e99.",
                    "evidence": [{"source_id": HIT.id, "quote": HIT.text}],
                }
            ]
        },
        {
            "claims": [
                {
                    "text": "Revenue is 125.0 EUR billions.",
                    "evidence": [{"source_id": HIT.id, "quote": HIT.text}],
                }
            ]
        },
        {"answer": "Uncited trailing prose", "claims": []},
        {"claims": "malformed"},
    ],
)
def test_unverifiable_claims_never_displayed(payload):
    result = answer(Retriever(), "Revenue?", generate=lambda *_: json.dumps(payload))
    assert result.status == "generation_failed" and not result.answer


def test_no_evidence_never_calls_generation_and_explicit_abstention():
    class Empty:
        def search(self, *args, **kwargs):
            return []

    def forbidden(*args):
        raise AssertionError("No evidence must not call a generator")

    assert answer(Empty(), "unknown", generate=forbidden).status == "no_evidence"
    assert (
        answer(Retriever(), "unknown", generate=lambda *_: '{"claims":[]}').status
        == "abstained"
    )


def test_source_bound_decimal_operations():
    operands = [
        Operand(HIT.id, HIT.text, v, "USD millions") for v in ["125.0", "100.0"]
    ]
    assert calculate("growth_percent", operands, [HIT])["value"] == "25.00"
    assert calculate("difference", operands, [HIT])["value"] == "25.0"
    assert calculate("ratio", operands, [HIT])["value"] == "1.25"
    negative = Operand(HIT.id, HIT.text, "(12.50)", "USD millions")
    assert calculate("sum", [operands[0], negative], [HIT])["value"] == "112.50"
    with pytest.raises(ValueError, match="absent"):
        calculate("sum", [Operand(HIT.id, HIT.text, "12.5", "USD millions")], [HIT])
    with pytest.raises(ValueError, match="Unit"):
        calculate("sum", [Operand(HIT.id, HIT.text, "125.0", "EUR")], [HIT])
    with pytest.raises(ValueError, match="Unsupported"):
        calculate('__import__("os")', operands, [HIT])
    zero = Hit("zero", "report", "text", 0, 5, "USD: 0", {}, 1)
    with pytest.raises(ValueError, match="zero"):
        calculate("ratio", [Operand("zero", zero.text, "0", "USD")] * 2, [zero])


def test_evidence_and_generation_budgets():
    result = answer(Retriever(), "Revenue?", max_evidence_chars=100)
    assert sum(len(s.text) for s in result.sources) <= 100
    assert result.sources[0].end == result.sources[0].start + len(
        result.sources[0].text
    )
    result = answer(Retriever(), "Revenue?", generate=lambda *_: "x" * 50001)
    assert result.status == "generation_failed"


def test_numeric_equivalence_preserves_currency_scale():
    payload = {
        "claims": [
            {
                "text": "Revenue is 125.00 USD million.",
                "evidence": [{"source_id": HIT.id, "quote": HIT.text}],
            }
        ]
    }
    assert (
        answer(Retriever(), "Revenue?", generate=lambda *_: json.dumps(payload)).status
        == "cited"
    )


def test_currency_symbols_are_not_assumed_to_be_us_dollars():
    payload = {
        "claims": [
            {
                "text": "Revenue is 125.0 $ billion.",
                "evidence": [{"source_id": HIT.id, "quote": HIT.text}],
            }
        ]
    }
    assert (
        answer(Retriever(), "Revenue?", generate=lambda *_: json.dumps(payload)).status
        == "generation_failed"
    )


@pytest.mark.parametrize(
    "claim",
    [
        "Dose: 50 g.",
        "Dose: 50 grams.",
        "Dose: 50grams.",
        "Dose: 50 mg/kg.",
        "Dose: 50 mcg.",
        "Balance: 100 EUR.",
        "Interval: 14 hours.",
    ],
)
def test_adjacent_number_unit_swaps_are_rejected(claim):
    from dataclasses import replace

    hit = replace(
        HIT,
        text="Dose: 50 mg. Mass: 5 g. Weight dose: 2 mg/kg. Trace: 3 mcg. Balance: 100 USD; tax: 200 EUR. Interval: 14 days; window: 2 hours.",
    )

    class Evidence:
        def search(self, *args, **kwargs):
            return [hit]

    payload = {
        "claims": [
            {"text": claim, "evidence": [{"source_id": hit.id, "quote": hit.text}]}
        ]
    }
    assert (
        answer(Evidence(), "dose", generate=lambda *_: json.dumps(payload)).status
        == "generation_failed"
    )


def test_medical_units_equivalence_and_compound_measurements():
    from dataclasses import replace

    hit = replace(
        HIT, text="Dose 50 mg, trace 3 µg, concentration 2 mg/mL, pressure 120 mmHg."
    )

    class Evidence:
        def search(self, *args, **kwargs):
            return [hit]

    for claim in [
        "Dose 50 mg.",
        "Dose 50 milligrams.",
        "Trace 3 mcg.",
        "Concentration 2 mg/mL.",
        "Pressure 120 mmHg.",
    ]:
        payload = {
            "claims": [
                {"text": claim, "evidence": [{"source_id": hit.id, "quote": hit.text}]}
            ]
        }
        assert (
            answer(
                Evidence(), "measurements", generate=lambda *_: json.dumps(payload)
            ).status
            == "cited"
        )


def test_scientific_decimal_values_and_exponent_span_precision():
    from decimal import Decimal

    hit = Hit(
        "scientific",
        "records",
        "row 2",
        0,
        40,
        "USD: 1e3; 1e-3; 1e100; 1e-100; 1e1001.",
        {},
        1,
    )
    operands = [Operand(hit.id, hit.text, value, "USD") for value in ["1e3", "1e-3"]]
    assert calculate("difference", operands, [hit])["value"] == "999.999"
    wide = [Operand(hit.id, hit.text, value, "USD") for value in ["1e100", "1e-100"]]
    result = calculate("sum", wide, [hit])
    assert result["precision"] >= 201
    value = Decimal(result["value"])
    assert value.as_tuple().digits[0] == value.as_tuple().digits[-1] == 1
    assert len(value.as_tuple().digits) == 201
    with pytest.raises(ValueError, match="oversized"):
        calculate("sum", [Operand(hit.id, hit.text, "1e1001", "USD")], [hit])
