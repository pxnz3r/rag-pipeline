"""Summarize paired numeric QA records without treating routes as independent cases."""

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


def wilson(correct, count):
    z = 1.959963984540054
    p, d = correct / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / d
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / d
    return [center - radius, center + radius]


def bootstrap(differences):
    rng = random.Random(20261003)
    samples = sorted(
        sum(rng.choices(differences, k=len(differences))) / len(differences)
        for _ in range(10000)
    )
    return [samples[249], samples[9749]]


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    grouped = {}
    for row in rows:
        key = row["route"]
        if row["id"] in grouped.setdefault(key, {}):
            raise ValueError("Duplicate case/route")
        grouped[key][row["id"]] = row
    if not rows:
        raise ValueError("No QA records")
    output = dict(
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cases=len({r["id"] for r in rows}),
        routes={},
        paired_route_differences={},
        caveats="Small numeric subset; common prompt and adapters, not a published upstream configuration. Wilson intervals are per route. Bootstrap differences resample paired cases. Persistent prompt cache affects latency; no SOTA or cross-domain conclusion.",
    )
    for route, records in sorted(grouped.items()):
        values = list(records.values())
        count = len(values)
        direct = sum(r["direct_correct"] for r in values)
        program = sum(r["program_correct"] for r in values)
        corrected = sum(
            r["program_correct"] and not r["direct_correct"] for r in values
        )
        harmed = sum(r["direct_correct"] and not r["program_correct"] for r in values)
        calls = [c for r in values for c in r["calls"]]
        output["routes"][route] = dict(
            cases=count,
            direct_correct=direct,
            direct_accuracy=direct / count,
            program_correct=program,
            program_accuracy=program / count,
            program_wilson95=wilson(program, count),
            corrected=corrected,
            harmed=harmed,
            program_minus_direct95=bootstrap(
                [int(r["program_correct"]) - int(r["direct_correct"]) for r in values]
            ),
            statuses=dict(Counter(r["program_status"] for r in values)),
            valid_but_wrong=sum(
                r["program_status"] == "calculated" and not r["program_correct"]
                for r in values
            ),
            generation_calls=len(calls),
            completion_tokens=sum(
                (c.get("usage") or {}).get("completion_tokens", 0) for c in calls
            ),
            cached_prompt_tokens=sum(
                (c.get("usage") or {})
                .get("prompt_tokens_details", {})
                .get("cached_tokens", 0)
                for c in calls
            ),
            prompt_tokens=sum(
                (c.get("usage") or {}).get("prompt_tokens", 0) for c in calls
            ),
            median_total_seconds=statistics.median(r["total_seconds"] for r in values),
            median_generation_seconds=statistics.median(c["seconds"] for c in calls),
            finish_reasons=dict(Counter(c["finish_reason"] for c in calls)),
        )
    routes = sorted(grouped)
    for pos, a in enumerate(routes):
        for b in routes[pos + 1 :]:
            ids = sorted(set(grouped[a]) & set(grouped[b]))
            differences = [
                int(grouped[b][i]["program_correct"])
                - int(grouped[a][i]["program_correct"])
                for i in ids
            ]
            if ids:
                output["paired_route_differences"][f"{b} minus {a}"] = dict(
                    paired_cases=len(ids),
                    mean=sum(differences) / len(ids),
                    bootstrap95=bootstrap(differences),
                )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.input), indent=2) + "\n")
