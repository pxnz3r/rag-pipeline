from __future__ import annotations

import argparse
import json
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
    p_ingest = sub.add_parser(
        "ingest", help="Extract local PDFs into a durable checkpoint."
    )
    p_ingest.add_argument("--base-dir", default=None)

    p_validate = sub.add_parser(
        "validate-notebook", help="Validate notebook schema and syntax."
    )
    p_validate.add_argument("path", nargs="?", default="Python3finale.ipynb")

    p_audit = sub.add_parser("audit-notebook", help="Run notebook pattern audit.")
    p_audit.add_argument("path", nargs="?", default="Python3finale.ipynb")

    p_bench = sub.add_parser("benchmark", help="Run pipeline benchmark harness.")
    p_bench.add_argument("--samples", type=int, default=20000)
    p_bench.add_argument("--repeats", type=int, default=5)
    p_bench.add_argument("--output", default="benchmarks/latest.json")

    p_smoke = sub.add_parser(
        "smoke",
        help="Run offline smoke by default; use --live for env-gated live smoke.",
    )
    p_smoke.add_argument("--live", action="store_true")

    args = parser.parse_args()
    root = _project_root()

    if args.command == "ingest":
        from .config import load_settings
        from .ingestion import ingest_pdfs

        try:
            settings = load_settings(args.base_dir)
            chunks = ingest_pdfs(settings)
            print(
                json.dumps(
                    {
                        "chunks": len(chunks),
                        "documents": len({c.pdf_name for c in chunks}),
                    }
                )
            )
            return 0
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"Ingestion failed: {exc}", file=sys.stderr)
            return 1

    if args.command == "test":
        return _run([sys.executable, "-m", "pytest"])

    if not (root / "scripts").is_dir():
        candidate = Path.cwd()
        if (candidate / "scripts").is_dir() and (
            candidate / "pyproject.toml"
        ).is_file():
            root = candidate
        elif args.command != "smoke":
            parser.error(
                "This developer command requires a rag-pipeline source checkout. Run it from the repository root."
            )

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
                "--repeats",
                str(args.repeats),
                "--output",
                str(args.output),
            ]
        )

    if args.command == "smoke":
        from .smoke import run_smoke

        result = run_smoke(live=args.live)
        print(
            json.dumps(
                {"mode": result.mode, "ok": result.ok, "message": result.message}
            )
        )
        return 0 if result.ok else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
