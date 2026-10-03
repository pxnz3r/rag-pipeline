"""Run real generation with shared numeric evidence and upstream control flows."""

import argparse
import ast
import hashlib
import json
import time
from decimal import Decimal
from pathlib import Path

from rag_pipeline import Index
from rag_pipeline.answers import _pack_evidence
from rag_pipeline.configuration import load_config, provider
from rag_pipeline.expressions import (
    EXPRESSION_SYSTEM,
    execute_expression,
    expression_schema,
)
from rag_pipeline.providers import ChatServer
from rag_pipeline.reasoning import (
    PROGRAM_SYSTEM,
    _decimal,
    execute_program,
    parse_program,
    program_evidence,
    program_schema,
    scoped_sources,
)

parser = argparse.ArgumentParser(
    description="Controlled numeric QA comparison; not an official FlashRAG or FinQA leaderboard run."
)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument(
    "--flashrag",
    type=Path,
    required=True,
    help="Trusted pinned upstream source checkout",
)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--runtime-manifest", type=Path)
parser.add_argument("--split", choices=["dev", "test"], default="dev")
parser.add_argument(
    "--routes",
    nargs="+",
    choices=["retrieval", "full-context", "iterative-2"],
    default=["retrieval", "full-context", "iterative-2"],
)
parser.add_argument("--limit", type=int, default=100)
parser.add_argument(
    "--program-format", choices=["steps", "expression"], default="steps"
)
args = parser.parse_args()
ROOT, FLASH = args.data, args.flashrag
settings = load_config(args.config)
BUDGET = settings.get("answer", {}).get("max_evidence_chars", 6500)
SYSTEM = (
    PROGRAM_SYSTEM.replace(
        'Return JSON {"steps":',
        'Return JSON with steps first and your numeric answer estimate last; format {"steps":',
    )
    .replace(
        "Never compute intermediate numbers yourself.",
        "The program must use original operands or earlier step references. After selecting the program, give your numeric answer estimate in the answer field as a string.",
    )
    .replace('return {"steps":[]}.', 'return {"steps":[],"answer":null}.')
)

if args.program_format == "expression":
    SYSTEM = (
        EXPRESSION_SYSTEM
        + " After the expression, give your numeric answer estimate as a string in the answer field (or null if insufficient)."
    )

CONFIG = {
    "device": "cpu",
    "save_retrieval_cache": False,
    "use_fid": False,
    "refiner_name": None,
}


class NoEvaluator:
    def __init__(self, *a):
        pass

    def evaluate(self, *a):
        raise RuntimeError("External evaluator must not run")


namespace = {"Evaluator": NoEvaluator}
for file, names in [
    ("pipeline/pipeline.py", {"BasicPipeline", "SequentialPipeline"}),
    ("pipeline/active_pipeline.py", {"IterativePipeline"}),
]:
    path = FLASH / "flashrag" / file
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class Data:
    def __init__(self, q):
        self.question = [q]

    def update_output(self, name, value):
        setattr(self, name, value)


class Prompt:
    def get_string(self, question, retrieval_result):
        return (question, retrieval_result)


class Generator:
    def __init__(self):
        self.client = provider(settings["generation"], "generation")
        if (
            not isinstance(self.client, ChatServer)
            or not self.client.structured_outputs
        ):
            raise ValueError(
                "Comparison requires explicit ChatServer structured outputs"
            )
        self.calls = []

    def generate(self, prompts):
        results = []
        for q, hits in prompts:
            sources = _pack_evidence(hits, BUDGET)
            payload = json.dumps(dict(question=q, evidence=program_evidence(sources)))
            started = time.perf_counter()
            response = self.client._request(
                "/chat/completions",
                self.client._payload(
                    SYSTEM,
                    payload,
                    expression_schema(sources, include_answer=True)
                    if args.program_format == "expression"
                    else program_schema(sources, include_answer=True, catalog=True),
                ),
                "Generation",
            )
            raw = response["choices"][0]["message"]["content"]
            self.calls.append(
                dict(
                    seconds=time.perf_counter() - started,
                    usage=response.get("usage"),
                    finish_reason=response["choices"][0].get("finish_reason"),
                    timings=response.get("timings"),
                    raw=raw,
                    source_ids=[h.id for h in sources],
                    evidence_chars=sum(len(h.text) for h in sources),
                    evidence=[h.to_dict() for h in sources],
                )
            )
            results.append(raw)
        return results


class Retriever:
    def __init__(self, index, record, doc, route):
        self.index, self.record, self.doc, self.route = index, record, doc, route
        metadata = json.loads(
            index.db.execute(
                "SELECT metadata FROM documents WHERE id=?", (doc,)
            ).fetchone()[0]
        )
        self.scope = {"report": record["filename"]}
        if "source_view" in metadata:
            self.scope["source_view"] = metadata["source_view"]

    def batch_search(self, questions):
        output = []
        for question in questions:
            if self.route == "full-context":
                hits = scoped_sources(self.index, self.scope, BUDGET)
            else:
                hits = self.index.search(
                    question,
                    filters=self.scope,
                    **settings["search"],
                )
            output.append(_pack_evidence(hits, BUDGET))
        return output


def numeric(value):
    if value is None:
        return None
    n, p = _decimal(str(value))
    return n / 100 if p else n


def correct(value, gold):
    return value is not None and Decimal(str(round(float(value), 5))) == Decimal(
        str(gold)
    )


def run(split, routes, limit):
    records = json.loads((ROOT / "records.json").read_text())[split]
    mapping = json.loads((ROOT / "mapping.json").read_text())
    results = []
    path = args.output
    already = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            already[(row["id"], row["route"])] = row
            results.append(row)
    with Index(
        ROOT / "index.sqlite", provider(settings["embedding"], "embedding")
    ) as index:
        for position, record in enumerate(records[:limit]):
            offset = position % len(routes)
            for route in routes[offset:] + routes[:offset]:
                if (record["id"], route) in already:
                    continue
                started = time.perf_counter()
                gen = Generator()
                ret = Retriever(index, record, mapping[record["id"]], route)
                kwargs = dict(
                    config=CONFIG,
                    prompt_template=Prompt(),
                    retriever=ret,
                    generator=gen,
                )
                pipeline = (
                    namespace["IterativePipeline"](iter_num=2, **kwargs)
                    if route == "iterative-2"
                    else namespace["SequentialPipeline"](**kwargs)
                )
                data = pipeline.run(Data(record["qa"]["question"]), do_eval=False)
                raw = data.pred[0]
                hits = data.retrieval_result[0]
                direct = program = None
                status = "invalid"
                steps = []
                try:
                    parsed = parse_program(raw)
                    try:
                        direct = numeric(parsed.get("answer"))
                    except (ValueError, ArithmeticError):
                        direct = None
                    if parsed.get("steps") == [] or (
                        args.program_format == "expression"
                        and parsed.get("expression") is None
                    ):
                        status = "abstained"
                    else:
                        result = (
                            execute_expression(parsed["expression"], hits)
                            if args.program_format == "expression"
                            else execute_program(dict(steps=parsed["steps"]), hits)
                        )
                        program = Decimal(result.value)
                        status = result.status
                        steps = result.steps
                except Exception as error:
                    status = type(error).__name__ + ":" + str(error)[:200]
                gold = record["qa"]["exe_ans"]
                row = dict(
                    id=record["id"],
                    split=split,
                    route=route,
                    direct_prediction=str(direct) if direct is not None else None,
                    program_prediction=str(program) if program is not None else None,
                    direct_correct=correct(direct, gold),
                    program_correct=correct(program, gold),
                    gold=gold,
                    program_status=status,
                    program_trace=steps,
                    calls=gen.calls,
                    total_seconds=time.perf_counter() - started,
                )
                with path.open("a") as out:
                    out.write(json.dumps(row) + "\n")
                results.append(row)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in [
                                "id",
                                "route",
                                "direct_prediction",
                                "program_prediction",
                                "gold",
                                "direct_correct",
                                "program_correct",
                                "program_status",
                                "total_seconds",
                            ]
                        }
                    ),
                    flush=True,
                )
    return results


if __name__ == "__main__":
    manifest = dict(
        protocol="finqa-numeric-common-program-v2",
        program_format=args.program_format,
        split=args.split,
        routes=args.routes,
        limit=args.limit,
        evidence_budget=BUDGET,
        route_order="counterbalanced cyclic rotation by record index; persistent model cache, cached tokens reported",
        search=settings["search"],
        embedding=settings["embedding"],
        runtime=json.loads(args.runtime_manifest.read_text())
        if args.runtime_manifest
        else None,
        prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
        generation={
            key: value
            for key, value in settings["generation"]["options"].items()
            if key
            in {
                "endpoint",
                "model",
                "temperature",
                "max_tokens",
                "json_mode",
                "structured_outputs",
                "timeout",
                "api_key_env",
                "token_limit_parameter",
            }
        },
        generation_options_sha256=hashlib.sha256(
            json.dumps(settings["generation"], sort_keys=True).encode()
        ).hexdigest(),
        sha256={
            str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT)
            else path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "records.json",
                ROOT / "mapping.json",
                ROOT / "index.sqlite",
                Path(__file__),
                *sorted(Path(__import__("rag_pipeline").__file__).parent.glob("*.py")),
                FLASH / "flashrag/pipeline/pipeline.py",
                FLASH / "flashrag/pipeline/active_pipeline.py",
            ]
        },
        upstream_scope="Actual unchanged AST-isolated BasicPipeline/SequentialPipeline/IterativePipeline classes, common custom adapters and prompt; no native published model configuration or official leaderboard score.",
    )
    manifest_path = args.output.with_suffix(".manifest.json")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Resume protocol mismatch; choose a new output file")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    run(args.split, args.routes, args.limit)
