import os
import argparse
from accelerate import Accelerator
import paper_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from print_utils import pprint
import torch
from tqdm import tqdm
from grpo_utils import generate_responses
import pandas as pd
from pathlib import Path

try:
    from reasoning.env import diagnose_completion, score_completions
except ModuleNotFoundError:
    from env import diagnose_completion, score_completions

DATASET = "data/paper_instructions_300K-v2"
batch_size = 4
data_size = 20
EVAL_TEMPERATURE = 0.2

# Pre-compile regex for post-processing
def post_process(response):
    if "<think>" in response and "</think>" not in response:
        response = response + "</think>"
    return response

def extract_answer_fast(response):
    think_end = response.find("</think>")
    if think_end != -1:
        return response[think_end + len("</think>"):].strip()
    return response.strip()


def extract_thinking_fast(response):
    match = thinking_pattern.search(response)
    if match:
        return match.group(1)
    else:
        match = fallback_pattern.search(response)
        return match.group(1).strip() if match else None

def print_stats(scores_list, output_path=None):
    df = pd.DataFrame(scores_list)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=4)
        print("Eval generations saved in: ", output_path)
    stats_cols = [col for col in df.columns if col not in ["question", "response", "answer"]]
    stats_df = df[stats_cols]
    pprint(stats_df.describe().round(2))
    match_rates = {
        name: float(df[name].mean())
        for name in ("datatype_matches", "schema_matches")
        if name in df
    }
    if match_rates:
        pprint(
            {
                name.replace("matches", "match_rate"): f"{value:.1%}"
                for name, value in match_rates.items()
            },
            title="Structural match rates",
            is_json=True,
        )
    return stats_df

def append_scores(total_scores, scores_dict):
    for k, v in scores_dict.items():
        if k not in total_scores:
            total_scores[k] = []

        total_scores[k].extend(v)
    return total_scores


def run_inference(
    llm,
    tokenizer,
    dataloader,
    max_new_tokens=200,
    output_path=None,
    temperature=EVAL_TEMPERATURE,
):
    print("Starting inference...")
    total_scores = {}
    # Merge the LoRA adapter for the eval decode loop (same ~1.6x speedup as
    # rollouts). Safe here: eval only generates + scores — no log_probs or
    # importance ratio — so the merged-vs-unmerged bf16 difference is irrelevant.
    # Guarded for the standalone __main__ path that loads a plain (non-PEFT) model.
    merged_for_eval = hasattr(llm, "merge_adapter")
    if merged_for_eval:
        llm.merge_adapter()
    for i, d in enumerate(tqdm(dataloader)):

        inputs = dict(
            input_ids = d["input_ids"],
            attention_mask = d["attention_mask"]
        )
        with torch.no_grad():
            outputs = generate_responses(
                llm,
                inputs,
                n_rollouts=1,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_length = d["input_ids"].shape[1]
        newly_generated_tokens = outputs[:, input_length:]
        out = tokenizer.batch_decode(newly_generated_tokens, skip_special_tokens=True)
        out = [post_process(o) for o in out] 
        model_num_tokens = (newly_generated_tokens != tokenizer.eos_token_id).to(torch.int32).sum(axis=-1).cpu().numpy()
        answers = [extract_answer_fast(o) for o in out]
        
        references = [item["answer"] for item in d["item"]]
        reward_batch = score_completions(
            out,
            references,
        )
        diagnostics = [
            diagnose_completion(completion, reference)
            for completion, reference in zip(out, references)
        ]
        
        stats = {} 
        stats["question"] = [item["prompt"] for item in d["item"]]
        stats["response"] = out
        stats["answer"] = answers
        stats["num_tokens"] = model_num_tokens
        stats["total_reward"] = reward_batch.total
        for k, v in reward_batch.components.items():
            stats[k] = v
        for name in (
            "datatype_matches",
            "schema_matches",
            "reference_datatype",
            "response_datatype",
        ):
            stats[name] = [diagnostic[name] for diagnostic in diagnostics]

        total_scores = append_scores(
            total_scores, stats
        )
    if merged_for_eval:
        llm.unmerge_adapter()
    stats_df = print_stats(total_scores, output_path=output_path)
    return stats_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paper GRPO inference.")
    parser.add_argument("model_name")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    # Load model with optimizations
    accelerator = Accelerator()

    llm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = paper_dataset.get_dataloader(
        DATASET,
        tokenizer=tokenizer,
        batch_size=batch_size,
        data_size=data_size,
        seed=4
    )
    
    llm, dataloader = accelerator.prepare(llm, dataloader)
    total_scores = run_inference(
        llm,
        tokenizer,
        dataloader,
        output_path=args.output_path,
    )
