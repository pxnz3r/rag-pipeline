"""Verify saved model responses against the release executor without new inference."""

import argparse
import hashlib
import json
from pathlib import Path

from rag_pipeline import Hit
from rag_pipeline.answers import _pack_evidence
from rag_pipeline.expressions import execute_expression
from rag_pipeline.reasoning import execute_program, parse_program


def replay(path, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("answer_only"):
        raise ValueError("Replay requires a generated program")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    mismatches = []
    for row in rows:
        call = row["calls"][-1]
        sources = [Hit(**h) for h in call["evidence"]]
        packed = _pack_evidence(sources, manifest["evidence_budget"])
        if [h.to_dict() for h in packed] != call["evidence"]:
            mismatches.append(
                dict(
                    id=row["id"], route=row["route"], failure="Evidence packing changed"
                )
            )
        value = None
        try:
            parsed = parse_program(call["raw"])
            expression = manifest.get("program_format") == "expression"
            if (
                parsed.get("steps") == []
                or expression
                and parsed.get("expression") is None
            ):
                status = "abstained"
            else:
                result = (
                    execute_expression(parsed["expression"], sources)
                    if expression
                    else execute_program(dict(steps=parsed["steps"]), sources)
                )
                value, status = result.value, result.status
        except Exception as error:
            status = type(error).__name__ + ":" + str(error)[:200]
        if (value, status) != (row["program_prediction"], row["program_status"]):
            mismatches.append(
                dict(
                    id=row["id"],
                    route=row["route"],
                    expected=[row["program_prediction"], row["program_status"]],
                    actual=[value, status],
                )
            )
    return dict(
        checked=len(rows),
        mismatches=mismatches,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        release_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(
                Path(__import__("rag_pipeline").__file__).parent.glob("*.py")
            )
        },
        scope="Deterministic packing/program replay, not a new model accuracy run or semantic correctness check.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = replay(args.input, args.manifest)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(checked=result["checked"], mismatches=len(result["mismatches"]))
        )
    )
    if result["mismatches"]:
        raise SystemExit(1)
