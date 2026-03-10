from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rag-pipeline", description="RAG pipeline utility CLI."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("test", help="Run pytest.")

    p_validate = sub.add_parser(
        "validate-notebook", help="Validate notebook schema and syntax."
    )
    p_validate.add_argument("path", nargs="?", default="Python3finale.ipynb")

    p_audit = sub.add_parser("audit-notebook", help="Run notebook pattern audit.")
    p_audit.add_argument("path", nargs="?", default="Python3finale.ipynb")

    p_bench = sub.add_parser("benchmark", help="Run pipeline benchmark harness.")
    p_bench.add_argument("--samples", type=int, default=20000)
    p_bench.add_argument("--output", default="benchmarks/latest.json")

    p_smoke = sub.add_parser(
        "smoke",
        help="Run offline smoke by default; use --live for env-gated live smoke.",
    )
    p_smoke.add_argument("--live", action="store_true")

    args = parser.parse_args()
    root = _project_root()

    if args.command == "test":
        return _run([sys.executable, "-m", "pytest"])

    if args.command == "validate-notebook":
        return _run(
            [
                sys.executable,
                str(root / "scripts" / "validate_notebook.py"),
                args.path,
            ]
        )

    if args.command == "audit-notebook":
        return _run(
            [
                sys.executable,
                str(root / "scripts" / "audit_notebook_patterns.py"),
                args.path,
            ]
        )

    if args.command == "benchmark":
        return _run(
            [
                sys.executable,
                str(root / "scripts" / "benchmark_pipeline.py"),
                "--samples",
                str(args.samples),
                "--output",
                str(args.output),
            ]
        )

    if args.command == "smoke":
        cmd = [sys.executable, str(root / "scripts" / "smoke_e2e.py")]
        if args.live:
            cmd.append("--live")
        return _run(cmd)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
