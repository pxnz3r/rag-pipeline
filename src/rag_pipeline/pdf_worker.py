"""Disposable PDF parser with OS resource caps on POSIX (not a security sandbox).

PDFium (Chrome's engine, through pypdfium2) replaced pypdf's layout mode: on
the 84 FinanceBench filings it extracted 16x faster, kept more gold evidence
numbers (99.9% vs 99.2%), opened the empty-password "encrypted" filings that
pypdf rejected, and produced a third of the text (no layout padding).
"""

import json
import sys
from pathlib import Path

MAX_PAGES = 3000
MAX_CHARS = 20 * 1024**2


def main():
    parser = sys.argv[3] if len(sys.argv) > 3 else "pdfium"
    # Layout models (Docling) need far more memory and CPU than PDFium.
    # Docling also writes its layout and table models to the cache on first use.
    memory, cpu, files = (
        (2 * 1024**3, 120, 64 * 1024**2)
        if parser == "pdfium"
        else (12 * 1024**3, 3600, 4 * 1024**3)
    )
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (files, files))
    except ImportError:  # Windows still has the parent's wall-clock timeout.
        pass
    if parser == "docling":
        pages = docling_pages(sys.argv[1])
    else:
        pages = pdfium_pages(sys.argv[1])
    if sum(map(len, pages)) > MAX_CHARS:
        raise ValueError("Too much extracted text")
    Path(sys.argv[2]).write_text(json.dumps(pages), encoding="utf-8")


def docling_pages(path):
    """Per-page Markdown: tables keep rows and columns, headings keep levels.

    OCR is off: born-digital filings carry text, and OCR would only add
    recognition errors and minutes per document.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    document = converter.convert(path).document
    if document.num_pages() > MAX_PAGES:
        raise ValueError("Too many pages")
    return [
        # Compact tables: column-width padding only spent chunk budget.
        document.export_to_markdown(page_no=number, compact_tables=True).replace(
            "\x00", ""
        )
        for number in range(1, document.num_pages() + 1)
    ]


def pdfium_pages(path):
    import pypdfium2 as pdfium

    # Opening with no password handles owner-only encryption (print or copy
    # restrictions), which most corporate filings carry; a user password fails.
    document = pdfium.PdfDocument(path)
    try:
        if len(document) > MAX_PAGES:
            raise ValueError("Too many pages")
        pages, chars = [], 0
        for number in range(len(document)):
            page = document[number]
            textpage = page.get_textpage()
            # NUL comes from broken font maps, not the document's text; offsets
            # refer to this stored extraction.
            text = (
                textpage.get_text_bounded()
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\x00", "")
            )
            textpage.close()
            page.close()
            chars += len(text)
            if chars > MAX_CHARS:
                raise ValueError("Too much extracted text")
            pages.append(text)
    finally:
        document.close()
    return pages


if __name__ == "__main__":
    main()
