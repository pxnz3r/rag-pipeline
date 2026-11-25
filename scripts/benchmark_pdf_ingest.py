"""Time lexical ingest of generated multi-page text PDFs (parsing cost, no models).

python scripts/benchmark_pdf_ingest.py --documents 48 --pages 20
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag_pipeline import Index


def pdf(path, pages, seed):
    writer = PdfWriter()
    font = DictionaryObject(
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
    for number in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = font
        lines = b"".join(
            b"0 -14 Td (Line %d of page %d in report %d: revenue USD %d.%d million) Tj "
            % (i, number, seed, seed * 7 + i, number)
            for i in range(45)
        )
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 10 Tf 40 760 Td " + lines + b"ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=48)
    parser.add_argument("--pages", type=int, default=20)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="rag-pdf-bench-") as directory:
        root = Path(directory) / "corpus"
        root.mkdir()
        for i in range(args.documents):
            pdf(root / f"report-{i:03d}.pdf", args.pages, i)
        with Index(Path(directory) / "index.sqlite") as index:
            started = time.perf_counter()
            stats = index.ingest(root)
            seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                "documents": args.documents,
                "pages": args.documents * args.pages,
                "chunks": stats["chunks"],
                "ingest_seconds": round(seconds, 2),
                "pages_per_second": round(args.documents * args.pages / seconds, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
