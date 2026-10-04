import json
import subprocess

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag_pipeline.sources import document, pdf_sections, spans


def test_csv_and_json_keep_numeric_labels(tmp_path):
    csv = tmp_path / "ledger.csv"
    csv.write_text(
        'account,USD millions,year\nRevenue,"1,234.50",2024\nLoss,(12.50),2024\n'
    )
    doc = document(csv, tmp_path)
    assert doc.sections[0] == (
        "row 2",
        "account: Revenue\nUSD millions: 1,234.50\nyear: 2024",
    )
    assert "(12.50)" in doc.sections[1][1]
    records = tmp_path / "accounts.json"
    records.write_text('[{"balance": 12345678901234567890.0123456789}]')
    assert (
        "12345678901234567890.0123456789" in document(records, tmp_path).sections[0][1]
    )
    csv.with_name(csv.name + ".meta.json").write_text(json.dumps({"company": "A"}))
    changed = document(csv, tmp_path)
    assert changed.fingerprint != doc.fingerprint
    assert not document(csv, tmp_path, changed.fingerprint).sections


@pytest.mark.parametrize("text", ["a,a\n1,2", "a,b\n1", "\n", "a,b\n1,2,3"])
def test_malformed_csv_fails(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        document(path, tmp_path)


def test_pdf_worker_real_parse_and_failures(tmp_path):
    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(blank)
    with pytest.raises(ValueError, match="OCR"):
        document(blank, tmp_path)
    # A real text-bearing PDF checks the subprocess, extraction and page locator.
    page = writer.add_blank_page(width=500, height=300)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 50 250 Td (Revenue USD 125.0 million) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(blank)
    extracted = document(blank, tmp_path)
    assert extracted.sections[0][0] == "page 2" and "125.0" in extracted.sections[0][1]
    with pytest.raises(ValueError, match="failed"):
        pdf_sections(b"not a pdf")


def test_worker_timeout_and_bounded_overlap(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 45)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ValueError, match="45 seconds"):
        pdf_sections(b"pdf")
    text = "clause exception; important number 125.0.\n" * 100
    offsets = list(spans(text, 180, 30))
    assert offsets[0][0] == 0 and offsets[-1][1] == len(text)
    assert all(0 < b - a <= 180 for a, b in offsets)
    assert all(
        next_a <= b and next_a > a for (a, b), (next_a, _) in zip(offsets, offsets[1:])
    )


def test_nul_text_fails_before_sqlite_window_truncation(tmp_path):
    path = tmp_path / "nul.txt"
    path.write_text("Revenue\x00USD 125 million")
    with pytest.raises(ValueError, match="NUL"):
        document(path, tmp_path)


def test_heading_hierarchy_ignores_fenced_code():
    from rag_pipeline.sources import contextual_spans

    text = (
        "# Parent\n## First\n"
        + "First passage. " * 30
        + "\n```python\n# Forged heading\n```\n"
        + "More first passage. " * 30
        + "\n## Second\n"
        + "Second passage. " * 30
    )
    result = list(contextual_spans(text, 100, 0, True))
    assert any(context == "Parent > First" for _, _, context in result)
    assert any(context == "Parent > Second" for _, _, context in result)
    assert all("Forged" not in context for _, _, context in result)
    assert all(text[a:b] and len(context) <= 300 for a, b, context in result)


@pytest.mark.parametrize("spelling", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_numbers_are_not_evidence(tmp_path, spelling):
    path = tmp_path / "invalid.json"
    path.write_text('{"value":' + spelling + "}")
    with pytest.raises(ValueError, match="Non-finite"):
        document(path, tmp_path)


def test_context_headings_stream_without_per_heading_retention():
    import tracemalloc

    from rag_pipeline.sources import contextual_spans

    text = "# Repeated heading\n" * 40000
    tracemalloc.start()
    try:
        result = list(contextual_spans(text, 900, 100, True))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result and all(context == "Repeated heading" for _, _, context in result)
    assert peak < len(text) * 4


def test_normalized_csv_and_json_expansion_is_bounded(tmp_path, monkeypatch):
    import rag_pipeline.sources as sources

    monkeypatch.setattr(sources, "MAX_BYTES", 256)
    path = tmp_path / "wide.csv"
    path.write_text("long_header_alpha,long_header_beta\n" + "1,2\n" * 20)
    assert path.stat().st_size < 256
    with pytest.raises(ValueError, match="Normalized extraction"):
        sources.document(path, tmp_path)
    path = tmp_path / "deep.json"
    path.write_text("[" * 20 + "[1,2,3,4,5,6,7,8,9,10]" + "]" * 20)
    assert path.stat().st_size < 256
    with pytest.raises(ValueError, match="Normalized extraction"):
        sources.document(path, tmp_path)
