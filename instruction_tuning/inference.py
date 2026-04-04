import argparse
import json
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """

parser = argparse.ArgumentParser(description="Batched inference eval on paper_instructions test split.")
parser.add_argument("--model_path", "-m", type=str, required=True)
parser.add_argument("--num_samples", "-n", type=int, default=1000)
parser.add_argument("--batch_size", "-bs", type=int, default=16)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--output_file", "-o", type=str, default=None)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

dataset = load_dataset("paperbd/paper_instructions_300K-v1", split="test")
dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

os.makedirs("instruction_tuning/evals", exist_ok=True)

if args.output_file is None:
    model_name = os.path.basename(os.path.normpath(args.model_path))
    output_path = f"instruction_tuning/evals/{model_name}_batched_results.jsonl"
else:
    output_path = f"instruction_tuning/evals/{args.output_file}"

def build_prompt(example):
    instruction = example["instruction"]
    inp = example.get("input", "")
    question = instruction if not inp else f"{instruction}\n\n{inp}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return question, tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

examples = list(dataset)
total = len(examples)
completed = 0

with open(output_path, "w") as f:
    for batch_start in range(0, total, args.batch_size):
        batch = examples[batch_start: batch_start + args.batch_size]

        questions, prompts, ground_truths = [], [], []
        for ex in batch:
            q, p = build_prompt(ex)
            questions.append(q)
            prompts.append(p)
            ground_truths.append(ex["output"])

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=4,
                repetition_penalty=1.1,
            )

        input_len = encoded["input_ids"].shape[1]
        for j, (q, gt) in enumerate(zip(questions, ground_truths)):
            generated = tokenizer.decode(
                output_ids[j][input_len:],
                skip_special_tokens=True,
            )
            record = {
                "id": batch_start + j,
                "question": q,
                "response": generated,
                "ground_truth": gt,
            }
            f.write(json.dumps(record) + "\n")
            print(f"[{batch_start + j + 1}/{total}] {generated[:80].replace(chr(10), ' ')}...")

        completed += len(batch)
        print(f"  batch {batch_start // args.batch_size + 1} done — {completed}/{total}")

print(f"\nSaved {total} results to {output_path}")
