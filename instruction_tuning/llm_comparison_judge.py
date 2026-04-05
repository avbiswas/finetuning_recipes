"""
Compare responses from 2+ models using an LLM judge.

Usage:
    uv run instruction_tuning/llm_comparison_judge.py \
        -i evals/model_a.jsonl evals/model_b.jsonl \
        -n model_a model_b
"""

import argparse
import asyncio
import json
import os
import random
from typing import Literal, get_args
from pydantic import BaseModel, Field, create_model
from openai import AsyncOpenAI
import outlines

JUDGE_MODEL = "openai/gpt-5.4-mini"
SEMAPHORE = 50

JUDGE_PROMPT = """\
You are an expert evaluator comparing AI assistant responses on research and academic questions.

Given a question, a ground truth reference, and {n} responses labelled {labels}, evaluate each response and pick the best one.

Scoring guide (1-5):
  1 = Very poor   2 = Poor   3 = Acceptable   4 = Good   5 = Excellent

Score each response on overall quality (faithfulness, correctness, relevance, completeness combined).
Then pick the single best response as the winner.

---
QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

{responses}
---

Provide brief reasoning (1-2 sentences), then scores and winner.
"""


def build_comparison_model(labels: list[str]):
    winner_type = Literal[tuple(labels)]
    score_fields = {f"score_{l}": (int, Field(..., ge=1, le=5)) for l in labels}
    return create_model(
        "ComparisonResult",
        reasoning=(
            str,
            Field(..., description="Brief reasoning for the scores and winner."),
        ),
        winner=(
            winner_type,
            Field(..., description=f"The best response: one of {labels}"),
        ),
        **score_fields,
    )


parser = argparse.ArgumentParser(
    description="LLM comparison judge across multiple eval files."
)
parser.add_argument(
    "--input_files",
    "-i",
    nargs="+",
    required=True,
    help="Two or more JSONL eval files to compare.",
)
parser.add_argument(
    "--model_names",
    "-m",
    nargs="+",
    default=None,
    help="Display names for each model (same order as --input_files). Defaults to filenames.",
)
parser.add_argument("--output_file", "-o", type=str, default=None)
parser.add_argument("--api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY"))
parser.add_argument("--limit", "-n", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if len(args.input_files) < 2:
    raise ValueError("Need at least 2 input files to compare.")
if not args.api_key:
    raise ValueError("Set OPENROUTER_API_KEY or pass --api_key")


def infer_model_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    for suffix in (
        "_mlx_results_judged",
        "_mlx_results",
        "_results_judged",
        "_results",
        "_batched_results",
    ):
        stem = stem.replace(suffix, "")
    return stem


model_names = args.model_names or [infer_model_name(f) for f in args.input_files]
assert len(model_names) == len(
    args.input_files
), "Number of model names must match input files."

if args.output_file is None:
    names_str = "_vs_".join(model_names)
    args.output_file = f"instruction_tuning/evals/comparison_{names_str}.jsonl"

# Load and index all files by id
all_records: dict[int, dict] = (
    {}
)  # id -> {question, ground_truth, responses: {model_name: response}}
for path, name in zip(args.input_files, model_names):
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rid = r["id"]
            if rid not in all_records:
                all_records[rid] = {
                    "id": rid,
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "responses": {},
                }
            all_records[rid]["responses"][name] = r["response"]

# Keep only records present in all models
complete = {
    rid: v for rid, v in all_records.items() if len(v["responses"]) == len(model_names)
}
records = sorted(complete.values(), key=lambda r: r["id"])

if args.limit:
    records = records[: args.limit]

total = len(records)
labels = [chr(65 + i) for i in range(len(model_names))]  # A, B, C...
ComparisonResult = build_comparison_model(labels)

client = AsyncOpenAI(api_key=args.api_key, base_url="https://openrouter.ai/api/v1")
model = outlines.from_openai(client, JUDGE_MODEL)

rng = random.Random(args.seed)


async def judge_one(sem: asyncio.Semaphore, idx: int, record: dict) -> dict:
    # Shuffle model order to avoid position bias; track label -> model_name mapping
    shuffled_names = list(model_names)
    rng.shuffle(shuffled_names)
    label_to_model = {label: name for label, name in zip(labels, shuffled_names)}

    responses_block = "\n\n".join(
        f"Response {label}:\n{record['responses'][label_to_model[label]]}"
        for label in labels
    )
    prompt = JUDGE_PROMPT.format(
        n=len(labels),
        labels=", ".join(labels),
        question=record["question"],
        ground_truth=record["ground_truth"],
        responses=responses_block,
    )

    async with sem:
        raw = await model(prompt, ComparisonResult, max_tokens=1024)
        result = (
            ComparisonResult.model_validate_json(raw) if isinstance(raw, str) else raw
        )

    winning_model = label_to_model[result.winner]
    scores = {label_to_model[l]: getattr(result, f"score_{l}") for l in labels}

    out = {
        "id": record["id"],
        "question": record["question"],
        "ground_truth": record["ground_truth"],
        "responses": record["responses"],
        "winner": winning_model,
        "scores": scores,
        "reasoning": result.reasoning,
        "label_order": label_to_model,
    }

    score_str = "  ".join(f"{name}={scores[name]}" for name in model_names)
    print(f"[{idx+1}/{total}] winner={winning_model}  {score_str}")
    return out


async def main():
    sem = asyncio.Semaphore(SEMAPHORE)
    tasks = [judge_one(sem, i, r) for i, r in enumerate(records)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda r: r["id"])

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    print("\n--- summary ---")
    win_counts = {name: 0 for name in model_names}
    avg_scores = {name: [] for name in model_names}
    for r in results:
        win_counts[r["winner"]] += 1
        for name in model_names:
            avg_scores[name].append(r["scores"][name])

    for name in model_names:
        wins = win_counts[name]
        avg = sum(avg_scores[name]) / len(avg_scores[name])
        print(
            f"  {name:40s}  wins={wins}/{total} ({100*wins/total:.1f}%)  avg_score={avg:.2f}"
        )

    print(f"\nSaved to {args.output_file}")


asyncio.run(main())
