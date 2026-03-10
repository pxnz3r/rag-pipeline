#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from rag_pipeline.smoke import run_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Run smoke checks for RAG pipeline.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run env-gated live prerequisite smoke (does not execute full paid API flow).",
    )
    args = parser.parse_args()

    result = run_smoke(live=args.live)
    payload = {"mode": result.mode, "ok": result.ok, "message": result.message}
    print(json.dumps(payload, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
