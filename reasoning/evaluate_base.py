import argparse
import json
import math
from pathlib import Path

try:
    from reasoning.grpo import (
        DOOM_LOOP_PENALTY,
        OUTPUT_FORMAT_MATCH_REWARD,
        OUTPUT_FORMAT_MISMATCH_REWARD,
        SEED,
        _extract_think_content,
        _has_doom_loop,
        _has_expected_format,
        _is_valid_json,
        _word_count,
        format_row,
    )
except ModuleNotFoundError:
    from grpo import (
        DOOM_LOOP_PENALTY,
        OUTPUT_FORMAT_MATCH_REWARD,
        OUTPUT_FORMAT_MISMATCH_REWARD,
        SEED,
        _extract_think_content,
        _has_doom_loop,
        _has_expected_format,
        _is_valid_json,
        _word_count,
        format_row,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate held-out rollouts and score the reasoning rewards."
    )
    parser.add_argument(
        "--model_path",
        "-m",
        default="paperbd/neuraltxt-135M-reasoning-base",
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        default="paperbd/paper_instructions_300K-v1",
    )
    parser.add_argument("--num_eval_rows", "-n", type=int, default=32)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def select_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_rollouts(rows: list[dict], reward_model) -> list[dict]:
    valid_indices = []
    valid_responses = []
    valid_references = []

    for index, row in enumerate(rows):
        completion = row["completion"]
        reference = row["reference"]
        if _has_expected_format(completion) and reference:
            _, response, _ = _extract_think_content(completion)
            valid_indices.append(index)
            valid_responses.append(response)
            valid_references.append(reference)

    raw_scores = [0.0] * len(rows)
    if valid_indices:
        scored = reward_model.batch_score(
            valid_responses,
            valid_references,
            batch_size=8,
        )
        for index, score in zip(valid_indices, scored):
            raw_scores[index] = float(score)

    results = []
    for index, row in enumerate(rows):
        completion = row["completion"]
        reference = row["reference"]
        reasoning, response, _ = _extract_think_content(completion)
        has_expected_format = _has_expected_format(completion)
        schema_matches = (
            bool(reference)
            and has_expected_format
            and _is_valid_json(reference) == _is_valid_json(response)
        )
        has_doom_loop = _has_doom_loop(completion)

        think_reward = 1.0 if has_expected_format else 0.0
        schema_reward = (
            OUTPUT_FORMAT_MATCH_REWARD
            if schema_matches
            else OUTPUT_FORMAT_MISMATCH_REWARD
        )
        doom_reward = DOOM_LOOP_PENALTY if has_doom_loop else 0.0

        raw_neuraltxt = raw_scores[index]
        length_multiplier = 1.0
        if raw_neuraltxt and reference:
            max_ok = max(_word_count(reference) * 1.5, _word_count(reference) + 20)
            response_words = _word_count(response)
            if response_words > max_ok:
                length_multiplier = max_ok / response_words
        neuraltxt_reward = raw_neuraltxt * length_multiplier

        rewards = {
            "think_format_reward": think_reward,
            "output_format_reward": schema_reward,
            "doom_loop_reward": doom_reward,
            "neuraltxt_reward": neuraltxt_reward,
        }
        results.append({
            **row,
            "reasoning": reasoning,
            "response": response,
            "diagnostics": {
                "has_expected_format": has_expected_format,
                "schema_matches": schema_matches,
                "has_doom_loop": has_doom_loop,
                "reference_is_json": _is_valid_json(reference),
                "response_is_json": _is_valid_json(response),
                "reference_words": _word_count(reference),
                "response_words": _word_count(response),
                "raw_neuraltxt_score": raw_neuraltxt,
                "length_multiplier": length_multiplier,
            },
            "rewards": rewards,
            "total_reward": sum(rewards.values()),
        })
    return results


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: list[dict]) -> dict:
    totals = [row["total_reward"] for row in results]
    reward_names = list(results[0]["rewards"]) if results else []
    reward_means = {
        name: mean([row["rewards"][name] for row in results])
        for name in reward_names
    }
    raw_neuraltxt = [
        row["diagnostics"]["raw_neuraltxt_score"]
        for row in results
    ]
    length_multipliers = [
        row["diagnostics"]["length_multiplier"]
        for row in results
    ]

    return {
        "num_rollouts": len(results),
        "mean_total_reward": mean(totals),
        "std_total_reward": (
            math.sqrt(mean([(value - mean(totals)) ** 2 for value in totals]))
            if totals
            else 0.0
        ),
        "min_total_reward": min(totals) if totals else 0.0,
        "max_total_reward": max(totals) if totals else 0.0,
        "mean_rewards": reward_means,
        "mean_raw_neuraltxt_score": mean(raw_neuraltxt),
        "mean_length_multiplier": mean(length_multipliers),
        "think_format_pass_rate": mean([
            float(row["diagnostics"]["has_expected_format"])
            for row in results
        ]),
        "schema_match_rate": mean([
            float(row["diagnostics"]["schema_matches"])
            for row in results
        ]),
        "doom_loop_rate": mean([
            float(row["diagnostics"]["has_doom_loop"])
            for row in results
        ]),
        "length_penalty_rate": mean([
            float(row["diagnostics"]["length_multiplier"] < 1.0)
            for row in results
        ]),
    }


def main():
    import torch
    from datasets import load_dataset
    from neuraltxt import NeuralTxtReward
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    args = parse_args()
    device = select_device(torch, args.device)
    model_name = args.model_path.rstrip("/").split("/")[-1]
    output_dir = Path(
        args.output_dir or f"reasoning/evals/{model_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = output_dir / "rollouts.jsonl"
    summary_path = output_dir / "summary.json"

    print(f"Loading model: {args.model_path}")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
    ).to(device)
    model.eval()

    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.shuffle(seed=args.seed)
    num_eval_rows = min(args.num_eval_rows, len(dataset))
    examples = list(dataset.select(range(num_eval_rows)))

    rollout_inputs = []
    for example_index, example in enumerate(examples):
        formatted = format_row(example)
        prompt_text = tokenizer.apply_chat_template(
            formatted["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for rollout_index in range(args.num_rollouts):
            rollout_inputs.append({
                "example_index": example_index,
                "rollout_index": rollout_index,
                "prompt": formatted["prompt"],
                "prompt_text": prompt_text,
                "reference": str(formatted["answer"]),
            })

    set_seed(args.seed)
    generated_rows = []
    for start in range(0, len(rollout_inputs), args.batch_size):
        batch = rollout_inputs[start : start + args.batch_size]
        encoded = tokenizer(
            [row["prompt_text"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        ).to(device)
        input_length = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        completions = tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True,
        )
        for row, completion in zip(batch, completions):
            generated_rows.append({
                **row,
                "completion": completion.strip(),
            })
        print(f"Generated {len(generated_rows)}/{len(rollout_inputs)}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("Loading NeuralTxt reward model")
    reward_model = NeuralTxtReward(backend="hf", device=device)
    results = score_rollouts(generated_rows, reward_model)
    summary = {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "num_eval_rows": num_eval_rows,
        "num_rollouts_per_example": args.num_rollouts,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        **summarize(results),
    }

    with rollouts_path.open("w", encoding="utf-8") as output_file:
        for row in results:
            output_file.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Rollouts: {rollouts_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
