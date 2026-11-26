"""Compare PDF text engines on FinanceBench: failures, CPU time, gold numbers kept.

    python scripts/prepare_financebench.py /tmp/fb
    python scripts/compare_pdf_extraction.py /tmp/fb

For each engine (pypdf layout, pypdf plain, PDFium) it extracts every filing
and reports the share of numbers from each question's gold evidence text that
appear on the extracted gold page(s).
"""

import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

S = Path(sys.argv[1])
manifest = json.loads((S / "financebench.json").read_text())
NUM = re.compile(r"\d[\d,]*\.?\d*")


def pages_pypdf(path, mode):
    from pypdf import PdfReader

    r = PdfReader(path)
    if r.is_encrypted:
        r.decrypt("")
    out = []
    for p in r.pages:
        t = p.extract_text(extraction_mode=mode) if "/Contents" in p else ""
        out.append(t.replace("\x00", ""))
    return out


def pages_pdfium(path):
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(path)
    out = []
    for i in range(len(doc)):
        page = doc[i]
        tp = page.get_textpage()
        out.append(tp.get_text_bounded().replace("\x00", "").replace("\r\n", "\n"))
        tp.close()
        page.close()
    doc.close()
    return out


def extract(args):
    engine, path = args
    t = time.perf_counter()
    try:
        if engine == "pdfium":
            pages = pages_pdfium(path)
        else:
            pages = pages_pypdf(path, engine)
        return engine, Path(path).name, pages, time.perf_counter() - t, None
    except Exception as e:  # noqa: BLE001
        return engine, Path(path).name, None, time.perf_counter() - t, repr(e)[:100]


docs = sorted({q["document"] for q in manifest["questions"]})
jobs = [(e, str(S / "corpus" / d)) for e in ("layout", "plain", "pdfium") for d in docs]
with ProcessPoolExecutor(4) as pool:
    results = list(pool.map(extract, jobs))
by = {(e, d): (p, t, err) for e, d, p, t, err in results}
for engine in ("layout", "plain", "pdfium"):
    seconds = sum(by[(engine, d)][1] for d in docs)
    failed = [d for d in docs if by[(engine, d)][2]]
    chars = sum(sum(map(len, by[(engine, d)][0] or [])) for d in docs)
    found = total = 0
    for q in manifest["questions"]:
        pages = by[(engine, q["document"])][0]
        if pages is None:
            total += 1
            continue
        text = " ".join(pages[p - 1] for p in q["evidence_pages"] if p - 1 < len(pages))
        squashed = re.sub(r"\s+", " ", text)
        nums = [n for n in NUM.findall(" ".join(q["evidence_text"])) if len(n) >= 3]
        if nums:
            total += 1
            found += sum(n in squashed for n in nums) / len(nums)
    print(
        json.dumps(
            {
                "engine": engine,
                "seconds": round(seconds, 1),
                "failed": failed,
                "chars_M": round(chars / 1e6, 1),
                "gold_number_coverage": round(found / total, 4),
            }
        )
    )
