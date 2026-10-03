"""Bounded local extraction; numbers and row labels are never rewritten by an LLM."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

FORMATS = {".pdf", ".txt", ".md", ".csv", ".json", ".jsonl", ".py", ".sql"}
MAX_BYTES = 20 * 1024 * 1024
KEY = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")


def metadata(values: dict) -> dict:
    if not isinstance(values, dict) or len(values) > 32:
        raise ValueError("Metadata must be an object with at most 32 fields")
    for key, value in values.items():
        if (
            not isinstance(key, str)
            or not KEY.fullmatch(key)
            or isinstance(value, bool)
            or not isinstance(value, (str, int))
            or len(str(value)) > 512
            or "\x00" in str(value)
        ):
            raise ValueError("Metadata requires short named string/integer fields")
    return values.copy()


@dataclass(frozen=True)
class Document:
    id: str
    fingerprint: str
    metadata: dict
    sections: list[tuple[str, str]]


def _nonfinite(value):
    raise ValueError("Non-finite JSON number: " + value)


def row_text(headers, values):
    """Deterministic column labels; numeric spellings are preserved verbatim."""
    if len(headers) != len(values):
        raise ValueError("Table row has the wrong column count")
    return "\n".join(
        f"{header}: {value}" if header else str(value)
        for header, value in zip(headers, values)
    )


def _bounded_json(obj, limit):
    parts, size = [], 0
    for part in json.JSONEncoder(ensure_ascii=False, indent=2).iterencode(obj):
        size += len(part.encode("utf-8"))
        if size > limit:
            raise ValueError("Normalized extraction exceeds 20 MiB")
        parts.append(part)
    return "".join(parts)


def _bounded_sections(pairs):
    result, size = [], 0
    for locator, text in pairs:
        size += len(text.encode("utf-8"))
        if size > MAX_BYTES:
            raise ValueError("Normalized extraction exceeds 20 MiB")
        result.append((locator, text))
    return result


def document(path: Path, root: Path, previous: str | None = None) -> Document:
    # Read one immutable snapshot so the fingerprint always describes parsed bytes.
    if path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError(f"Unsafe or oversized source: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Source exceeds byte limit")
    sidecar = path.with_name(path.name + ".meta.json")
    meta = {}
    if sidecar.exists() or sidecar.is_symlink():
        if sidecar.is_symlink() or sidecar.stat().st_size > 32768:
            raise ValueError("Unsafe or oversized metadata")
        meta = metadata(json.loads(sidecar.read_text(encoding="utf-8")))
    doc_id = path.relative_to(root).as_posix()
    meta.setdefault("title", path.stem)
    fingerprint = hashlib.sha256(
        raw + json.dumps(meta, sort_keys=True).encode()
    ).hexdigest()
    if fingerprint == previous:
        return Document(doc_id, fingerprint, meta, [])
    if path.suffix.lower() == ".pdf":
        sections = pdf_sections(raw)
    else:
        text = raw.decode("utf-8-sig", errors="strict")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            rows = csv.reader(io.StringIO(text))
            header = next(rows, [])
            if (
                not header
                or len(set(header)) != len(header)
                or any(not h for h in header)
            ):
                raise ValueError("CSV requires unique nonempty column headers")
            sections, extracted_bytes = [], 0
            for row_number, row in enumerate(rows, 2):
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(f"CSV row {row_number} has the wrong column count")
                rendered = row_text(header, row)
                extracted_bytes += len(rendered.encode("utf-8"))
                if extracted_bytes > MAX_BYTES:
                    raise ValueError("Normalized extraction exceeds 20 MiB")
                sections.append((f"row {row_number}", rendered))
        elif suffix in {".json", ".jsonl"}:
            objects = (
                [
                    json.loads(
                        line, parse_float=str, parse_int=str, parse_constant=_nonfinite
                    )
                    for line in text.splitlines()
                    if line.strip()
                ]
                if suffix == ".jsonl"
                else json.loads(
                    text, parse_float=str, parse_int=str, parse_constant=_nonfinite
                )
            )
            objects = objects if isinstance(objects, list) else [objects]
            # Explicit keys, original numeric spellings for JSON string values, and
            # stable row locators. JSON numeric lexical spelling may be normalized.
            sections, extracted_bytes = [], 0
            for i, obj in enumerate(objects, 1):
                rendered = _bounded_json(obj, MAX_BYTES - extracted_bytes)
                extracted_bytes += len(rendered.encode("utf-8"))
                sections.append((f"record {i}", rendered))
        else:
            sections = [("text", text)]
    sections = _bounded_sections(sections)
    if any("\x00" in text for _, text in sections):
        raise ValueError(
            "Extracted text contains NUL characters; convert the source first"
        )
    if not sections or not any(text.strip() for _, text in sections):
        raise ValueError(f"No extractable text in {path.name}; scanned PDFs need OCR")
    return Document(doc_id, fingerprint, meta, sections)


def pdf_sections(raw: bytes) -> list[tuple[str, str]]:
    with tempfile.TemporaryDirectory(prefix="rag-pdf-") as directory:
        source, output = Path(directory) / "input.pdf", Path(directory) / "output.json"
        source.write_bytes(raw)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rag_pipeline.pdf_worker",
                    str(source),
                    str(output),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=45,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("PDF extraction exceeded 45 seconds") from exc
        if (
            result.returncode
            or not output.exists()
            or output.stat().st_size > MAX_BYTES
        ):
            raise ValueError("PDF extraction failed or exceeded resource limits")
        return [
            (f"page {i}", text)
            for i, text in enumerate(json.loads(output.read_text()), 1)
            if text.strip()
        ]


def spans(text: str, size: int = 900, overlap: int = 100):
    """Original character offsets, paragraph/line boundaries, bounded overlap."""
    if (
        any(not isinstance(v, int) or isinstance(v, bool) for v in (size, overlap))
        or size < 100
        or not 0 <= overlap < size
    ):
        raise ValueError("Chunk size must be >=100 and overlap smaller than size")
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + size // 2, end),
                text.rfind(". ", start + size // 2, end),
            )
            if boundary > start:
                end = boundary + 1
        if text[start:end].strip():
            yield start, end
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def headings(text):
    stack, fence = [], None
    for line_match in re.finditer(r"[^\n]*(?:\n|$)", text):
        line = line_match[0]
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            a, b = marker.span(1)
            symbol, width = line[a], b - a
            if fence is None:
                fence = (symbol, width)
            elif symbol == fence[0] and width >= fence[1]:
                fence = None
        if fence is None and (
            match := re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$", line)
        ):
            level, title = (
                len(match[1]),
                re.sub(r"[ \t]+#+[ \t]*$", "", match[2]).strip()[:300],
            )
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            yield (
                line_match.start(),
                level,
                title,
                " > ".join(t for _, t in stack)[:300],
            )


def contextual_spans(text: str, size: int, overlap: int, enabled: bool):
    """Extractive Markdown headings only; fence contents never become headings.

    Offsets and evidence remain original. Context is bounded indexing guidance,
    not an LLM summary or independently citable evidence.
    """
    if not enabled:
        for a, b in spans(text, size, overlap):
            yield a, b, ""
        return

    records = headings(text)
    pending, context = next(records, None), ""
    for a, b in spans(text, size, overlap):
        while pending is not None and pending[0] <= a:
            context = pending[3]
            pending = next(records, None)
        yield a, b, context
