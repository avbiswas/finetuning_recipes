"""Generate rollouts from MLX reasoning checkpoints and score with NeuralTxtReward."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reasoning.env import (
    SEED,
    SYSTEM_PROMPT as REASONING_SYSTEM_PROMPT,
    _extract_think_content,
    _has_doom_loop,
    _has_expected_format,
    _is_valid_json,
    _word_count,
    format_row,
)

SFT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLX checkpoints with NeuralTxtReward.")
    parser.add_argument("--ckpt_dirs", nargs="*", default=[
        "models/mlx/run3_checkpoint-96",
        "models/mlx/run3_checkpoint-2560",
        "models/mlx/run3_checkpoint-4096",
    ])
    parser.add_argument("--sft_model", default="paperbd/neuraltxt-v1-135M-mlx")
    parser.add_argument("--dataset_name", default="paperbd/paper_instructions_300K-v1")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", default="reasoning/evals/run3_checkpoints")
    return parser.parse_args()


def short_name(path: str) -> str:
    return Path(path).name


def generate_mlx(model_path, examples, use_reasoning_prompt, args, output_dir, label):
    from mlx_lm import load, batch_generate
    from mlx_lm.sample_utils import make_sampler

    name = short_name(model_path)
    print(f"\n=== {label} ({name}) ===", flush=True)
    print(f"Loading: {model_path}", flush=True)

    model, tokenizer = load(model_path)
    print(f"  model loaded", flush=True)

    prompts = []
    references = []
    for ex in examples:
        if use_reasoning_prompt:
            formatted = format_row(ex)
            prompt_text = tokenizer.apply_chat_template(
                formatted["prompt"], tokenize=False, add_generation_prompt=True,
            )
            ref = str(formatted["answer"])
        else:
            instruction = ex["instruction"]
            inp = ex.get("input", "")
            question = instruction if not inp else f"{instruction}\n\n{inp}"
            messages = [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            ref = str(ex["output"])
        prompts.append(tokenizer.encode(prompt_text))
        references.append(ref)

    sampler = make_sampler(temp=args.temperature) if args.temperature > 0 else make_sampler(temp=0.0)
    print(f"  generating {len(prompts)} responses (batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens})...", flush=True)
    t_start = time.time()

    completions = []
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        result = batch_generate(
            model, tokenizer, prompts=batch_prompts,
            max_tokens=args.max_new_tokens, sampler=sampler, verbose=False,
        )
        completions.extend([t.strip() for t in result.texts])
        n_done = len(completions)
        elapsed = time.time() - t_start
        eta = (elapsed / n_done) * (len(prompts) - n_done)
        print(f"  {n_done}/{len(prompts)} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)

    gen_path = output_dir / f"generations_{name}.jsonl"
    rows = []
    with gen_path.open("w", encoding="utf-8") as f:
        for i, (ref, comp) in enumerate(zip(references, completions)):
            row = {"id": i, "reference": ref, "completion": comp}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"  saved to {gen_path}", flush=True)
    return name, rows


def score_rollouts(rows, reward_model, use_reasoning):
    """Score rollouts with NeuralTxtReward, extracting answers from <think> tags."""
    valid_indices = []
    valid_responses = []
    valid_references = []

    for i, row in enumerate(rows):
        completion = row["completion"]
        reference = row["reference"]
        if use_reasoning:
            if _has_expected_format(completion) and reference:
                _, response, _ = _extract_think_content(completion)
                valid_indices.append(i)
                valid_responses.append(response)
                valid_references.append(reference)
        else:
            if completion and reference:
                valid_indices.append(i)
                valid_responses.append(completion)
                valid_references.append(reference)

    raw_scores = [0.0] * len(rows)
    if valid_indices:
        scored = reward_model.batch_score(valid_responses, valid_references, batch_size=8)
        for idx, score in zip(valid_indices, scored):
            raw_scores[idx] = float(score)

    results = []
    for i, row in enumerate(rows):
        completion = row["completion"]
        reference = row["reference"]
        reasoning, response, _ = _extract_think_content(completion) if use_reasoning else ("", completion, 0)
        has_fmt = _has_expected_format(completion) if use_reasoning else True
        schema_ok = (
            bool(reference) and has_fmt
            and _is_valid_json(reference) == _is_valid_json(response)
        ) if use_reasoning else None
        has_dl = _has_doom_loop(completion)

        raw_nt = raw_scores[i]
        length_mul = 1.0
        if raw_nt and reference:
            max_ok = max(_word_count(reference) * 1.5, _word_count(reference) + 20)
            rw = _word_count(response)
            if rw > max_ok:
                length_mul = max_ok / rw
        length_penalty = raw_nt * (length_mul - 1.0) if raw_nt and length_mul < 1.0 else 0.0
        nt_score = raw_nt + 0.5 * length_penalty  # length_penalty_weight = 0.5

        results.append({
            **row,
            "reasoning": reasoning,
            "response": response,
            "has_expected_format": has_fmt,
            "schema_matches": schema_ok,
            "has_doom_loop": has_dl,
            "raw_neuraltxt_score": raw_nt,
            "neuraltxt_reward": nt_score,
            "length_multiplier": length_mul,
            "reference_words": _word_count(reference),
            "response_words": _word_count(response),
        })
    return results


def summarize(results, model_name):
    n = len(results)
    nt_scores = [r["neuraltxt_reward"] for r in results]
    raw_scores = [r["raw_neuraltxt_score"] for r in results]
    mean_nt = sum(nt_scores) / n if n else 0.0
    mean_raw = sum(raw_scores) / n if n else 0.0

    fmt_pass = sum(1 for r in results if r["has_expected_format"]) / n if n else 0.0
    schema_match = sum(1 for r in results if r.get("schema_matches")) / n if n else 0.0
    doom_rate = sum(1 for r in results if r["has_doom_loop"]) / n if n else 0.0
    len_pen_rate = sum(1 for r in results if r["length_multiplier"] < 1.0) / n if n else 0.0
    valid_count = sum(1 for r in results if r["response"] and r["reference"])
    mean_resp_words = sum(r["response_words"] for r in results) / n if n else 0.0

    return {
        "model": model_name,
        "num_samples": n,
        "mean_neuraltxt_score": mean_nt,
        "mean_raw_neuraltxt_score": mean_raw,
        "think_format_pass_rate": fmt_pass,
        "schema_match_rate": schema_match,
        "doom_loop_rate": doom_rate,
        "length_penalty_rate": len_pen_rate,
        "valid_response_count": valid_count,
        "mean_response_words": mean_resp_words,
    }


def main():
    from datasets import load_dataset
    from neuraltxt import NeuralTxtReward

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading test split: {args.dataset_name} (test, n={args.num_samples})", flush=True)
    dataset = load_dataset(args.dataset_name, split="test")
    dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
    examples = list(dataset)

    all_summaries = []

    # --- Reasoning checkpoints ---
    for ckpt_dir in args.ckpt_dirs:
        name, rows = generate_mlx(
            ckpt_dir, examples, use_reasoning_prompt=True,
            args=args, output_dir=output_dir, label="Reasoning ckpt",
        )
        all_summaries.append(("reasoning", name, rows))

    # --- SFT baseline ---
    name_sft, rows_sft = generate_mlx(
        args.sft_model, examples, use_reasoning_prompt=False,
        args=args, output_dir=output_dir, label="SFT baseline",
    )
    all_summaries.append(("sft", name_sft, rows_sft))

    # --- Score with NeuralTxtReward ---
    print("\n=== Scoring with NeuralTxtReward ===", flush=True)
    reward_model = NeuralTxtReward(backend="hf")

    scored = {}
    for model_type, name, rows in all_summaries:
        print(f"Scoring {name}...", flush=True)
        use_reasoning = (model_type == "reasoning")
        results = score_rollouts(rows, reward_model, use_reasoning)
        summary = summarize(results, name)

        # Save results
        results_path = output_dir / f"scored_{name}.jsonl"
        with results_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        scored[name] = {"results": results, "summary": summary, "type": model_type}
        print(f"  {name}: mean NeuralTxt = {summary['mean_neuraltxt_score']:.4f} "
              f"(raw: {summary['mean_raw_neuraltxt_score']:.4f}), "
              f"fmt_pass: {summary['think_format_pass_rate']:.1%}, "
              f"doom: {summary['doom_loop_rate']:.1%}", flush=True)

    # --- Head-to-head comparisons ---
    print("\n=== Head-to-Head Comparisons ===", flush=True)
    sft_name = name_sft
    sft_nt_scores = [r["neuraltxt_reward"] for r in scored[sft_name]["results"]]

    comparisons = []
    for model_type, name, _ in all_summaries:
        if name == sft_name:
            continue
        ckpt_nt = [r["neuraltxt_reward"] for r in scored[name]["results"]]
        wins = ties = losses = 0
        diffs = []
        for a, b in zip(ckpt_nt, sft_nt_scores):
            diff = a - b
            diffs.append(diff)
            if diff > 0.001:
                wins += 1
            elif diff < -0.001:
                losses += 1
            else:
                ties += 1
        n = len(ckpt_nt)
        mean_diff = sum(diffs) / n
        comparisons.append({
            "model": name,
            "model_mean_nt": scored[name]["summary"]["mean_neuraltxt_score"],
            "sft_mean_nt": scored[sft_name]["summary"]["mean_neuraltxt_score"],
            "mean_delta": mean_diff,
            "wins": wins, "losses": losses, "ties": ties,
            "win_rate": wins / n, "loss_rate": losses / n,
            "tie_rate": ties / n,
        })

    # --- Final summary ---
    full_summary = {
        "dataset": args.dataset_name,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "models": {},
        "head_to_head_vs_sft": comparisons,
    }
    for model_type, name, _ in all_summaries:
        full_summary["models"][name] = scored[name]["summary"]

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(full_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"\n=== Final Results ===", flush=True)
    print(json.dumps(full_summary, indent=2, ensure_ascii=True), flush=True)
    print(f"\nSaved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
