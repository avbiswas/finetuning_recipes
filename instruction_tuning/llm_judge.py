import argparse
import asyncio
import json
import os
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import outlines

JUDGE_MODEL = "openai/gpt-5.4-mini"
SEMAPHORE = 50

JUDGE_PROMPT = """\
You are an expert evaluator for AI assistant responses on research and academic questions.

Given a question, the model's response, and a ground truth reference answer, score the response on 4 dimensions from 1 to 5.

Scoring guide:
  1 = Very poor   2 = Poor   3 = Acceptable   4 = Good   5 = Excellent

Dimensions:
- faithfulness:        Does the response contain only factually correct claims? Penalise hallucinations.
- answer_correctness:  How closely does the response match the ground truth semantically?
- relevance:           Does the response directly address what was asked, without padding or going off-topic?
- completeness:        Does the response cover the key points from the ground truth without omitting important details?

Provide a brief reasoning (1-2 sentences) before the scores.

---
QUESTION:
{question}

MODEL RESPONSE:
{response}

GROUND TRUTH:
{ground_truth}
---
"""


class JudgeScore(BaseModel):
    reasoning: str = Field(..., description="Brief explanation of the scores.")
    faithfulness: int = Field(..., ge=1, le=5)
    answer_correctness: int = Field(..., ge=1, le=5)
    relevance: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)


parser = argparse.ArgumentParser(description="Async LLM judge for eval JSONL files.")
parser.add_argument("--input_file", "-i", type=str, required=True)
parser.add_argument("--output_file", "-o", type=str, default=None)
parser.add_argument("--api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY"))
parser.add_argument("--limit", "-n", type=int, default=None)
args = parser.parse_args()

if not args.api_key:
    raise ValueError("Set OPENROUTER_API_KEY or pass --api_key")

if args.output_file is None:
    args.output_file = args.input_file.replace(".jsonl", "_judged.jsonl")

client = AsyncOpenAI(
    api_key=args.api_key,
    base_url="https://openrouter.ai/api/v1",
)
model = outlines.from_openai(client, JUDGE_MODEL)


async def judge_one(sem: asyncio.Semaphore, idx: int, total: int, record: dict) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=record["question"],
        response=record["response"],
        ground_truth=record["ground_truth"],
    )
    async with sem:
        raw = await model(prompt, JudgeScore, max_tokens=4000)
        score = JudgeScore.model_validate_json(raw) if isinstance(raw, str) else raw

    result = {
        **record,
        "scores": {
            "faithfulness": score.faithfulness,
            "answer_correctness": score.answer_correctness,
            "relevance": score.relevance,
            "completeness": score.completeness,
        },
        "reasoning": score.reasoning,
    }
    avg = sum(result["scores"].values()) / 4
    print(
        f"[{idx+1}/{total}] "
        f"F={score.faithfulness} AC={score.answer_correctness} "
        f"R={score.relevance} C={score.completeness} "
        f"avg={avg:.2f} | {record['response'][:60].replace(chr(10), ' ')}..."
    )
    return result


async def main():
    with open(args.input_file) as f:
        records = [json.loads(line) for line in f]

    if args.limit:
        records = records[:args.limit]

    total = len(records)
    sem = asyncio.Semaphore(SEMAPHORE)

    tasks = [judge_one(sem, i, total, r) for i, r in enumerate(records)]
    results = await asyncio.gather(*tasks)

    # preserve original order
    results.sort(key=lambda r: r["id"])

    with open(args.output_file, "w") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")

    keys = ["faithfulness", "answer_correctness", "relevance", "completeness"]
    print("\n--- summary ---")
    for k in keys:
        vals = [r["scores"][k] for r in results]
        print(f"  {k:22s}  avg={sum(vals)/len(vals):.2f}  min={min(vals)}  max={max(vals)}")
    overall = [sum(r["scores"].values()) / 4 for r in results]
    print(f"  {'overall':22s}  avg={sum(overall)/len(overall):.2f}")
    print(f"\nSaved to {args.output_file}")


asyncio.run(main())
