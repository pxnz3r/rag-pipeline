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
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    except ImportError:  # Windows still has the parent's wall-clock timeout.
        pass
    import pypdfium2 as pdfium

    # Opening with no password handles owner-only encryption (print or copy
    # restrictions), which most corporate filings carry; a user password fails.
    document = pdfium.PdfDocument(sys.argv[1])
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
    Path(sys.argv[2]).write_text(json.dumps(pages), encoding="utf-8")


if __name__ == "__main__":
    main()
