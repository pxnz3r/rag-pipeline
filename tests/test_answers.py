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
