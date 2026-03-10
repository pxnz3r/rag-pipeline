from __future__ import annotations

import re
import unicodedata


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufffd", " ").replace("\x00", " ")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def is_text_garbled(text: str, threshold: float = 0.40) -> bool:
    if not text:
        return True
    total = len(text)
    if total == 0:
        return True

    junk = 0
    for ch in text:
        if ch in "\n\r\t ":
            continue
        if ch == "\ufffd" or ch == "\x00":
            junk += 1
            continue
        if unicodedata.category(ch).startswith("C"):
            junk += 1
    return (junk / total) > threshold
