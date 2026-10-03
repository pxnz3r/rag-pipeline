from decimal import Decimal

import pytest

from rag_pipeline import Hit
from rag_pipeline.expressions import execute_expression


def test_formula_compiles_and_rejects_code_or_fabricated_values():
    text = "Revenue rose from 100 to 125."
    sources = [Hit("s", "report", "row", 0, len(text), text, {}, 0)]
    assert execute_expression("([N1]-[N0])/[N0]", sources).value == "0.25"
    assert execute_expression("(125[N1]-100[N0])/100[N0]", sources).value == "0.25"
    assert execute_expression("125 [N1] − 100 [N0]", sources).value == "25"
    with pytest.raises(ValueError, match="mismatch"):
        execute_expression("999[N1]", sources)
    result = execute_expression("(N1-N0)/N0", sources)
    assert Decimal(result.value) == Decimal(".25") and result.sources == sources
    assert execute_expression("mean(N0,N1)", sources).value == "112.5"
    assert execute_expression("(N0+N1)/2", sources).value == "112.5"
    assert execute_expression("revenue[N1]/previous[N0]", sources).value == "1.25"
    assert execute_expression("(125-100)/100", sources).value == "0.25"
    assert Decimal(execute_expression("-N0", sources).value) == -100
    for expression in [
        '__import__("os").system("id")',
        "N0.__class__",
        "N0[0]",
        "N0+123",
        "True+N0",
        "N0/zero",
        "one+one",
        "mean(*[N0,N1])",
        "mean(value=N0)",
    ]:
        with pytest.raises(ValueError):
            execute_expression(expression, sources)


def test_ambiguous_literals_and_resource_limits():
    text = "Value 125; other value 125; previous 100."
    sources = [Hit("s", "report", "row", 0, len(text), text, {}, 0)]
    with pytest.raises(ValueError, match="ambiguous"):
        execute_expression("125-N2", sources)
    assert execute_expression("N0-N2", sources).value == "25"
    for value in ["(" * 500 + "N0" + ")" * 500, "N0+" * 40 + "N0", "N0" * 2001]:
        with pytest.raises(ValueError):
            execute_expression(value, sources)


def test_expression_generation_uses_same_grounded_executor():
    from rag_pipeline import reason_from_sources

    text = "Revenue 100 then 125."
    sources = [Hit("s", "report", "row", 0, len(text), text, {}, 0)]
    result = reason_from_sources(
        "Relative growth?",
        sources,
        generate=lambda system, evidence: '{"expression":"(N1-N0)/N0"}',
        program_format="expression",
    )
    assert result.value == "0.25" and result.attempts == 1
    assert (
        reason_from_sources(
            "Growth?",
            sources,
            generate=lambda *a: '{"expression":null}',
            program_format="expression",
        ).status
        == "abstained"
    )
    assert (
        reason_from_sources(
            "Growth?",
            sources,
            generate=lambda *a: '{"expression":"N1","extra":1}',
            program_format="expression",
        ).status
        == "program_failed"
    )
    with pytest.raises(ValueError, match="format"):
        reason_from_sources(
            "Growth?", sources, generate=lambda *a: "", program_format="python"
        )
