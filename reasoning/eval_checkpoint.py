"""Evaluate a LoRA-finetuned checkpoint vs the base model."""
import argparse
import json
import math
from pathlib import Path

try:
    from reasoning.env import SEED, format_row, score_rollouts, summarize
except ModuleNotFoundError:
    from env import SEED, format_row, score_rollouts, summarize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA checkpoint vs the reasoning-base model."
    )
    parser.add_argument(
        "--adapter_path",
        default="models/reasoning_models/checkpoint-25",
    )
    parser.add_argument(
        "--base_model",
        default="paperbd/neuraltxt-135M-reasoning-base",
    )
    parser.add_argument(
        "--dataset_name",
        default="paperbd/paper_instructions_300K-v1",
    )
    parser.add_argument("--num_eval_rows", type=int, default=32)
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


def main():
    import torch
    from datasets import load_dataset
    from neuraltxt import NeuralTxtReward
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    args = parse_args()
    device = select_device(torch, args.device)
    adapter_name = Path(args.adapter_path).name
    output_dir = Path(
        args.output_dir or f"reasoning/evals/{adapter_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = output_dir / "rollouts.jsonl"
    summary_path = output_dir / "summary.json"

    print(f"Loading base model: {args.base_model}")
    print(f"Loading adapter: {args.adapter_path}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
    ).to(device)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.merge_and_unload()
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

    del model, base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("Loading NeuralTxt reward model")
    reward_model = NeuralTxtReward(backend="hf", device=device)
    results = score_rollouts(generated_rows, reward_model)
    summary = {
        "model_path": str(args.adapter_path),
        "base_model": args.base_model,
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
