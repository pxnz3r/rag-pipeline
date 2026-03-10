#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import nbformat


def sanitize_notebook_code(src: str) -> str:
    lines = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            lines.append("pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def validate_notebook(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    missing_ids = [i for i, cell in enumerate(raw.get("cells", [])) if "id" not in cell]
    if missing_ids:
        raise ValueError(f"Cells missing id fields: {missing_ids}")

    nb = nbformat.read(path.open("r", encoding="utf-8"), as_version=4)
    nbformat.validate(nb)

    syntax_errors = []
    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if not src.strip():
            continue
        try:
            ast.parse(sanitize_notebook_code(src))
        except SyntaxError as exc:
            syntax_errors.append((idx, exc.lineno, exc.offset, exc.msg))

    if syntax_errors:
        details = ", ".join(
            f"cell={idx}:line={line}:col={col}:{msg}"
            for idx, line, col, msg in syntax_errors
        )
        raise SyntaxError(f"Notebook syntax check failed: {details}")

    print(f"Notebook validation passed: {path}")


def main() -> int:
    notebook_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Python3finale.ipynb")
    )
    validate_notebook(notebook_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
