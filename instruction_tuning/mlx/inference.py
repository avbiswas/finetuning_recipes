import argparse
import json
import os
from datasets import load_dataset
from mlx_lm import load, batch_generate

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """

parser = argparse.ArgumentParser(description="MLX batched eval on paper_instructions test split.")
parser.add_argument("--model_path", "-m", type=str, required=True)
parser.add_argument("--num_samples", "-n", type=int, default=1000)
parser.add_argument("--batch_size", "-bs", type=int, default=16)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--output_file", "-o", type=str, default=None)
args = parser.parse_args()

model, tokenizer = load(args.model_path)

dataset = load_dataset("paperbd/paper_instructions_300K-v1", split="test")
dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

os.makedirs("instruction_tuning/evals", exist_ok=True)

if args.output_file is None:
    model_name = os.path.basename(os.path.normpath(args.model_path))
    output_path = f"instruction_tuning/evals/{model_name}_mlx_results.jsonl"
else:
    output_path = f"instruction_tuning/evals/{args.output_file}"

examples = list(dataset)
total = len(examples)

def build_prompt(example):
    instruction = example["instruction"]
    inp = example.get("input", "")
    question = instruction if not inp else f"{instruction}\n\n{inp}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.encode(prompt_str)
    return question, token_ids

completed = 0

with open(output_path, "w") as f:
    for batch_start in range(0, total, args.batch_size):
        batch = examples[batch_start: batch_start + args.batch_size]

        questions, prompts, ground_truths = [], [], []
        for ex in batch:
            q, token_ids = build_prompt(ex)
            questions.append(q)
            prompts.append(token_ids)
            ground_truths.append(ex["output"])

        result = batch_generate(
            model,
            tokenizer,
            prompts=prompts,
            max_tokens=args.max_new_tokens,
            verbose=False,
        )

        for j, (q, gt, response) in enumerate(zip(questions, ground_truths, result.texts)):
            record = {
                "id": batch_start + j,
                "question": q,
                "response": response,
                "ground_truth": gt,
            }
            f.write(json.dumps(record) + "\n")
            print(f"[{batch_start + j + 1}/{total}] {response[:80].replace(chr(10), ' ')}...")

        completed += len(batch)
        print(f"  batch {batch_start // args.batch_size + 1} done — {completed}/{total}")

print(f"\nSaved {total} results to {output_path}")
