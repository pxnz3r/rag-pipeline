"""Disposable PDF parser with OS resource caps on POSIX (not a security sandbox)."""

import json
import logging
import sys
from pathlib import Path


def main():
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (40, 40))
        resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024**2, 20 * 1024**2))
    except ImportError:  # Windows still has the parent's wall-clock timeout.
        pass
    from pypdf import PdfReader

    logging.disable(logging.CRITICAL)
    reader = PdfReader(sys.argv[1])
    if reader.is_encrypted or len(reader.pages) > 1000:
        raise ValueError("Encrypted PDF or too many pages")
    pages, chars = [], 0
    for page in reader.pages:
        text = (
            page.extract_text(extraction_mode="layout") if "/Contents" in page else ""
        )
        chars += len(text)
        if chars > 10 * 1024**2:
            raise ValueError("Too much extracted text")
        pages.append(text)
    Path(sys.argv[2]).write_text(json.dumps(pages), encoding="utf-8")


if __name__ == "__main__":
    main()
