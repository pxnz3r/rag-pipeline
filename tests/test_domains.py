import json
import subprocess
import sys

import pytest

from rag_pipeline import Index, answer
from rag_pipeline.domains import citations, get, registry
from rag_pipeline.navigation import Navigation


def run(*args, cwd, env=None):
    return subprocess.run(
        [sys.executable, "-m", "rag_pipeline.cli", *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **(env or {})},
    )


def test_builtin_packs_and_unknown_domain():
    assert {"finance", "law", "healthcare", "trading", "general"} <= set(registry())
    assert get("healthcare").local_only and not get("finance").local_only
    assert "published" in get("trading").metadata
    with pytest.raises(ValueError, match="Unknown domain"):
        get("astrology")


def test_healthcare_keeps_every_stage_local_unless_network_is_allowed(tmp_path):
    keys = {"VOYAGE_API_KEY": "x", "ANTHROPIC_API_KEY": "y"}
    local = json.loads(
        run("--domain", "healthcare", "doctor", cwd=tmp_path, env=keys).stdout
    )
    assert local["capabilities"]["network"] is False
    assert local["capabilities"]["services"] == []
    allowed = json.loads(
        run(
            "--domain",
            "healthcare",
            "doctor",
            cwd=tmp_path,
            env={**keys, "RAG_ALLOW_NETWORK": "1"},
        ).stdout
    )
    assert allowed["capabilities"]["network"] is True


def test_legal_citations_carry_exact_offsets(tmp_path):
    text = (
        "Claims arise under 42 U.S.C. § 1983 and 17 C.F.R. 240.10b-5; see "
        "Miranda v. Arizona, 384 U.S. 436 (1966), and Section § 2.3(b)."
    )
    found = citations(text)
    assert [c["citation"] for c in found] == [
        "42 U.S.C. § 1983",
        "17 C.F.R. 240.10b-5",
        "384 U.S. 436",
        "§ 2.3(b)",
    ]
    assert all(text[c["start"] : c["end"]] == c["citation"] for c in found)
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "brief.txt").write_text(text)
    with Index(tmp_path / "i.sqlite") as index:
        index.ingest(root)
        result = Navigation(index).citations("brief.txt", "text")
        assert result["citations"][0]["citation"] == "42 U.S.C. § 1983"


def test_guidance_reaches_the_generation_prompt(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "r.txt").write_text("Revenue was USD 125.0 million in 2024.")
    prompts = []

    def generate(system, evidence):
        prompts.append(system)
        return '{"claims": []}'

    with Index(tmp_path / "i.sqlite") as index:
        index.ingest(root)
        answer(index, "revenue", generate=generate, guidance=get("finance").guidance)
    assert "Domain guidance: Financial filings" in prompts[0]


def test_sec_companyfacts_become_concept_sections(tmp_path):
    facts = {
        "cik": 66740,
        "entityName": "3M COMPANY",
        "facts": {
            "us-gaap": {
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "label": "Payments to Acquire Property, Plant, and Equipment",
                    "units": {
                        "USD": [
                            {
                                "start": "2018-01-01",
                                "end": "2018-12-31",
                                "val": 1577000000,
                                "fy": 2018,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2019-02-07",
                            }
                        ]
                    },
                }
            }
        },
    }
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "3m.json").write_text(json.dumps(facts))
    with Index(tmp_path / "i.sqlite") as index:
        index.ingest(root)
        hit = index.search("payments property plant equipment 2018")[0]
    assert hit.locator == "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
    assert "2018-01-01 to 2018-12-31" in hit.text and "1577000000" in hit.text
