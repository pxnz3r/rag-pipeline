"""Compare exact evidence output with trusted historical code; no model inference."""

import argparse
import ast
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

from rag_pipeline import Hit, reasoning


def measure(old_revision, sizes, repeats):
    source = subprocess.check_output(
        ["git", "show", old_revision + ":src/rag_pipeline/reasoning.py"]
    )
    namespace = vars(reasoning).copy()
    functions = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"numeric_catalog", "program_evidence"}
    ]
    if len(functions) != 2:
        raise ValueError("Historical implementation missing")
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            "trusted historical source",
            "exec",
        ),
        namespace,
    )
    result = dict(
        scope="Synthetic long unbroken numeric line; preprocessing only, not retrieval/generation latency or accuracy.",
        old_revision=old_revision,
        old_source_sha256=hashlib.sha256(source).hexdigest(),
        new_source_sha256=hashlib.sha256(
            Path(reasoning.__file__).read_bytes()
        ).hexdigest(),
        repeats=repeats,
        cases=[],
    )
    for size in sizes:
        text = " ".join(str(i) for i in range(size))
        hits = [Hit("s", "book", "text", 0, len(text), text, {}, 0)]
        old = namespace["program_evidence"](hits)
        new = reasoning.program_evidence(hits)
        if old != new or namespace["numeric_catalog"](
            hits
        ) != reasoning.numeric_catalog(hits):
            raise ValueError("Evidence/provenance changed")
        times = {"old": [], "new": []}
        for i in range(repeats):
            for name in ["old", "new"] if i % 2 == 0 else ["new", "old"]:
                started = time.perf_counter()
                (
                    namespace["program_evidence"]
                    if name == "old"
                    else reasoning.program_evidence
                )(hits)
                times[name].append((time.perf_counter() - started) * 1000)
        old_ms, new_ms = (
            statistics.median(times["old"]),
            statistics.median(times["new"]),
        )
        result["cases"].append(
            dict(
                numbers=size,
                original_chars=len(text),
                annotated_sha256=hashlib.sha256(
                    json.dumps(new, sort_keys=True).encode()
                ).hexdigest(),
                identical_output=True,
                old_median_ms=old_ms,
                new_median_ms=new_ms,
                ratio=old_ms / new_ms,
            )
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-revision", required=True, help="Trusted audited local Git revision"
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[10000, 50000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or any(not 1 <= n <= 100000 for n in args.sizes):
        parser.error("Invalid measurement budget")
    result = measure(args.old_revision, args.sizes, args.repeats)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["cases"], indent=2))
