from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
import argparse
import json
import paper_dataset
import torch
from prompt_utils import system_prompt 
from grpo_utils import (
    calculate_entropy,
    calculate_grpo_loss,
    calculate_kld_loss,
)
from rollout import calculate_log_probs, collect_rollouts
import numpy as np
from print_utils import pprint
from peft import get_peft_model, LoraConfig, AutoPeftModelForCausalLM
import yaml
from accelerate import Accelerator
import random
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from inference import run_inference
from tqdm import tqdm
import math
import gc
from pathlib import Path

try:
    from reasoning.env import score_completions
except ModuleNotFoundError:
    from env import score_completions

parser = argparse.ArgumentParser(description="Train GRPO on paper instructions.")
parser.add_argument("config_file")
parser.add_argument(
    "-o",
    "--output_model_id",
    required=True,
    help="Output model id under models/.",
)
args = parser.parse_args()

config_file = args.config_file
output_model_dir = Path("models") / args.output_model_id
logs_dir = output_model_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# Load configuration
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# Extract hyperparameters from config
model_name = config["model"]["name"]
rollout_batch_size = config["training"]["rollout_batch_size"]
batch_size = config["training"]["batch_size"]
n_rollouts = config["training"]["n_rollouts"]
max_new_tokens = config["model"]["max_new_tokens"]
dataset_name_or_path = config["data"]["dataset"]
gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
learning_rate = config["training"]["learning_rate"]
num_epochs = config["training"]["num_epochs"]
log_every = config["training"]["log_every"]
train_data_size = config["data"]["train_data_size"]
eval_data_size = config["data"].get(
    "eval_data_size",
    config["data"].get("test_data_size", 512),
)
test_batch_size = config["data"]["test_batch_size"]
kld_weight = config["loss"]["kld_weight"]
entropy_weight = config["loss"]["entropy_weight"]
loss_implementation = config["loss"].get("loss_implementation", "dr_grpo")
# dr_grpo's constant denominator; defaults to the generation length. Acts as an
# lr reparametrization, so retune learning_rate if you change it.
dr_grpo_max_tokens = config["loss"].get("max_tokens", max_new_tokens)
top_p = config["training"]["top_p"]
temperature = config["training"]["temperature"]
# Eval generation uses a separate, lower temperature: eval is a measurement /
# checkpoint-selection signal, so it wants low variance, not exploration.
eval_temperature = config["training"].get("eval_temperature", 0.2)
from_sft = config["model"].get("from_sft", False)
buffer_size = config["training"].get("buffer_size", 500)
num_repeats = config["training"].get("num_repeats", 5)
eval_every_train_step = config["training"].get("eval_every_train_step", 8)
std_normalize_advantages = config["training"].get("std_normalize_advantages", False)

if n_rollouts < 2:
    raise ValueError("GRPO requires n_rollouts >= 2 so group advantages are non-zero.")

accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

if from_sft:
    # Load the existing LoRA model directly
    llm = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        is_trainable=True,
        attn_implementation="sdpa",
    )
else:
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    lora_config = LoraConfig(
        task_type="CAUSAL_LM", 
        r=32, 
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )

    llm = get_peft_model(llm, lora_config)

if kld_weight > 0:
    if from_sft:
        ref_llm = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            is_trainable=False,
            attn_implementation="sdpa",
        )
    else:
        ref_llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    ref_llm.eval()
    for param in ref_llm.parameters():
        param.requires_grad_(False)
else:
    ref_llm = None

llm.print_trainable_parameters()
llm = accelerator.prepare(llm)
if ref_llm is not None:
    ref_llm = accelerator.prepare(ref_llm)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

train_dataset_seed = accelerator.process_index
test_dataset_seed = 16

dataloader = paper_dataset.get_dataloader(
    dataset_name_or_path,
    tokenizer=tokenizer,
    batch_size=rollout_batch_size,
    split=config["data"].get("train_split", "train"),
    data_size=train_data_size,
    seed=train_dataset_seed,
)
test_dataloader = paper_dataset.get_dataloader(
    dataset_name_or_path,
    tokenizer=tokenizer,
    batch_size=test_batch_size,
    split=config["data"].get("test_split", "test"),
    data_size=eval_data_size,
    seed=test_dataset_seed,
)

optimizer = torch.optim.Adam(llm.parameters(), lr=learning_rate)
optimizer, dataloader, test_dataloader = accelerator.prepare(
    optimizer, dataloader, test_dataloader
)

print("All loaded!!")

def save_model(postfix=""):
    unwrapped_llm = accelerator.unwrap_model(llm)
    model_dir = output_model_dir / f"llm{postfix}"
    unwrapped_llm.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def inference(csv_suffix=""):
    eval_path = logs_dir / f"eval_generations_{csv_suffix}.json"
    stats_df = run_inference(
        accelerator.unwrap_model(llm),
        tokenizer,
        test_dataloader,
        max_new_tokens=max_new_tokens,
        output_path=eval_path,
        temperature=eval_temperature,
    )
    summary = stats_df.describe().round(6).to_dict()
    with (logs_dir / f"eval_summary_{csv_suffix}.json").open("w") as f:
        json.dump(jsonable(summary), f, indent=2, allow_nan=False)
    return stats_df["total_reward"].mean(), stats_df["total_reward"].std()


def write_responses_to_file(responses, batch_idx, items):
    with (logs_dir / "train_responses.txt").open("a") as f:
        for i, response in enumerate(responses):
            item = items[i]
            answer = item["answer"]
            answer = f"Ground Truth: {answer}" if answer is not None else ""

            f.write(
                f"Batch {batch_idx}, Response {i}:\n{response}\n{'='*50}\n{answer}\n{'='*50}\n"
            )


def write_logs(log):
    log_str = " ".join([f"{k}={v}" for k, v in log.items()])
    with (logs_dir / "train_metrics.txt").open("a") as f:
        f.write(log_str + "\n")
    with (logs_dir / "train_metrics.jsonl").open("a") as f:
        f.write(json.dumps(jsonable(log), ensure_ascii=True, allow_nan=False) + "\n")


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_values(values):
    values = np.array(values, dtype=np.float32)
    return {
        "mean": float(np.mean(values)) if len(values) else 0.0,
        "std": float(np.std(values)) if len(values) else 0.0,
        "min": float(np.min(values)) if len(values) else 0.0,
        "max": float(np.max(values)) if len(values) else 0.0,
    }


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


class GRPO:
    def __init__(self):
        self.reset()
        self.num_experiences = 0
        self.num_training = 0

    def reset(self):
        self.buffer = []
        self.losses = []
        self.rewards = []
        self.individual_rewards = defaultdict(list)

    def log(self):
        mean_loss = np.mean(self.losses)
        mean_reward = np.mean(self.rewards)
        std_reward = np.std(self.rewards)

        individual_rewards = {k: np.mean(v) for k, v in self.individual_rewards.items()}
        rewards_breakdown_str = "\n".join(
            [
                f"[blue]{k}: [/blue] [bold]{v:.3f}[/bold]"
                for k, v in individual_rewards.items()
            ]
        )
        pprint(
            f"""
[blue]training steps:[/blue]: [bold]{self.num_training}[/bold]
[blue]num experiences:[/blue]: [bold]{self.num_experiences}[/bold]
[blue]loss:[/blue]: [bold]{mean_loss:.3f}[/bold]
[blue]reward:[/blue]: [bold]{mean_reward:.3f} +- {std_reward:.3f}[/bold]
{rewards_breakdown_str}
[green]inference scores:[/green]: [bold]{self.mean_inference_score:.3f} +- {self.std_inference_score:.3f}[/bold]
""",
            title="",
        )
        write_logs(
            dict(
                training_steps=self.num_training,
                num_experiences=self.num_experiences,
                mean_loss=mean_loss,
                mean_reward=mean_reward,
                std_reward=std_reward,
                inference_score=self.mean_inference_score,
                reward_breakdown={
                    name: summarize_values(values)
                    for name, values in self.individual_rewards.items()
                },
                **{
                    f"{name}_mean": value
                    for name, value in individual_rewards.items()
                },
            )
        )

    def train(self):
        optimizer.zero_grad()
        i = 0
        num_train_events = 0
        best_model_id = 0
        best_inference_score = -np.inf
        for epoch in range(num_epochs):
            for batch in dataloader:
                experiences = self.collect_experiences(batch, i)
                self.num_experiences += len(experiences)
                self.buffer.extend(experiences)

                i += 1
                if i % log_every == 0 and len(self.buffer) > 0:
                    pprint(
                        f"{accelerator.process_index}, [bold green]batch {i=}, buffer length: {len(self.buffer)}, Rewards: {np.mean(self.rewards):.3f} +- {np.std(self.rewards):.3f}[/bold green]"
                    )

                if len(self.buffer) >= buffer_size:
                    print("Will be training now!")
                    self.train_on_buffer()

                    clear_gpu_memory()

                    num_train_events += 1
                    self.buffer = []

                    # Only run the (expensive) full eval + checkpoint logic every
                    # Nth train event. num_train_events is incremented above, so
                    # the first event (1) is skipped and eval fires at 8, 16, ...
                    # (<=0 disables the throttle -> eval every event, as before).
                    if accelerator.is_main_process and (
                        eval_every_train_step <= 0
                        or num_train_events % eval_every_train_step == 0
                    ):
                        self.mean_inference_score, self.std_inference_score = inference(
                            num_train_events
                        )

                        save_model()
                        if self.mean_inference_score > best_inference_score:
                            pprint(
                                f"New best inference score: {self.mean_inference_score:.3f}"
                            )
                            save_model(f"_best_{best_model_id}")
                            best_model_id += 1
                            best_inference_score = self.mean_inference_score

                        self.log()

                    clear_gpu_memory()

                    self.reset()

    def calculate_logits(self, full_responses, attention_masks):
        token_log_probs, _ = calculate_log_probs(
            llm,
            full_responses,
            attention_masks,
        )
        return token_log_probs

    def collect_experiences(self, batch, i):
        llm.eval()
        with torch.inference_mode():
            rollouts = collect_rollouts(
                accelerator.unwrap_model(llm),
                tokenizer,
                batch,
                max_new_tokens=max_new_tokens,
                n_rollouts=n_rollouts,
                top_p=top_p,
                temperature=temperature,
                # Reuse the training batch size to chunk the old_log_probs forward:
                # the training forward over this many sequences already fits (and is
                # heavier), so the rollout forward over the same chunk is safe.
                logprob_chunk_size=batch_size,
            )

        reward_batch = score_completions(
            rollouts.completion_texts,
            rollouts.references,
        )

        # advantages = [num_examples, n_rollouts]
        # Mean-center always. std-normalization is opt-in via config: when off
        # (default), advantage magnitude tracks real reward separation instead of
        # amplifying the residual (judge noise + length) in near-degenerate groups
        # to unit scale. When on, this is classic GRPO normalization.
        rewards = reward_batch.total.reshape(rollouts.num_prompts, rollouts.group_size)
        advantages = rewards - np.mean(rewards, axis=1, keepdims=True)
        if std_normalize_advantages:
            advantages = advantages / (
                np.std(rewards, axis=1, keepdims=True) + 1e-8
            )
        advantages = advantages.reshape(-1, 1)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        self.rewards.extend(rewards.flatten().tolist())
        for reward_type, reward_value in reward_batch.components.items():
            self.individual_rewards[reward_type].extend(reward_value.flatten().tolist())

        advantages = advantages.cpu()
        return rollouts.to_experiences(advantages)

    def train_on_buffer(self):
        accelerator.wait_for_everyone()
        llm.train()
        random.shuffle(self.buffer)
        self.buffer = self.buffer[:buffer_size]
        total_examples = len(self.buffer)
        optimizer.zero_grad()
        needs_training = False
        num_steps = 0

        total_steps = math.ceil(total_examples / batch_size) * num_repeats
        progress_bar = tqdm(
            range(total_steps), desc="Training", disable=not accelerator.is_main_process
        )
        for _ in range(num_repeats):
            for i in range(0, total_examples, batch_size):
                with accelerator.accumulate(llm):
                    training_batch = self.buffer[i : i + batch_size]
                    self.num_training += 1
                    num_steps += 1
                    loss = self.train_on_batch(training_batch)
                    accelerator.backward(loss)
                    needs_training = True
                    optimizer.step()
                    optimizer.zero_grad()
                    needs_training = False
                    if accelerator.is_main_process:
                        progress_bar.set_description(
                            f"loss: {np.mean(self.losses):.3f}"
                        )
                        progress_bar.update(1)
        if needs_training:
            optimizer.step()
            optimizer.zero_grad()
        accelerator.wait_for_everyone()
        llm.eval()

    def train_on_batch(self, batch):
        input_ids = pad_sequence(
            [x[0] for x in batch],
            batch_first=True,
            padding_side="left",
            padding_value=tokenizer.pad_token_id,
        ).to(accelerator.device)

        attention_masks = pad_sequence(
            [torch.ones_like(x[0]) for x in batch],
            batch_first=True,
            padding_side="left",
            padding_value=0,
        ).to(accelerator.device)

        response_masks = pad_sequence(
            [x[1] for x in batch],
            batch_first=True,
            padding_side="left",
            padding_value=0,
        ).to(accelerator.device)

        old_log_probs = pad_sequence(
            [x[2] for x in batch],
            batch_first=True,
            padding_side="left",
            padding_value=0,
        ).to(accelerator.device)

        advantages = (
            torch.cat([x[3] for x in batch], dim=0).unsqueeze(-1).to(accelerator.device)
        )

        log_probs, full_log_probs = calculate_log_probs(
            llm,
            input_ids,
            attention_masks,
        )
        reasoning_loss = calculate_grpo_loss(
            log_probs,
            old_log_probs,
            advantages,
            response_masks,
            loss_implementation=loss_implementation,
            max_tokens=dr_grpo_max_tokens,
        )

        total_loss = reasoning_loss

        if kld_weight > 0:
            with torch.no_grad():
                _, ref_log_probs = calculate_log_probs(
                    ref_llm,
                    input_ids,
                    attention_masks,
                )
            kld_loss = calculate_kld_loss(
                full_log_probs,
                ref_log_probs,
                response_masks,
                loss_implementation=loss_implementation,
                max_tokens=dr_grpo_max_tokens,
            )
            total_loss = total_loss + kld_weight * kld_loss

        if entropy_weight > 0:
            entropy = calculate_entropy(
                full_log_probs,
                response_masks,
                loss_implementation=loss_implementation,
                max_tokens=dr_grpo_max_tokens,
            )
            total_loss = total_loss - entropy_weight * entropy

        self.losses.append(total_loss.item())
        return total_loss


if __name__ == "__main__":
    GRPO().train()
