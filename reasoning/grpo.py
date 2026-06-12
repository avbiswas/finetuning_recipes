import argparse
import json
import math
import re
from pathlib import Path

from neuraltxt import NeuralTxtReward

SEED = 3407
OUTPUT_FORMAT_MATCH_REWARD = 0.5
OUTPUT_FORMAT_MISMATCH_REWARD = -0.5
DOOM_LOOP_PENALTY = -1.0

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
Generate your reasoning first inside <think> and </think> tags. After </think>, generate only the requested final response.
When a structured format such as JSON is requested, the content after </think> must contain only that format, without Markdown fences or additional commentary.
    """

_reward_model: NeuralTxtReward | None = None


def _get_reward_model() -> NeuralTxtReward:
    global _reward_model
    if _reward_model is None:
        _reward_model = NeuralTxtReward(backend="hf")
    return _reward_model


def _extract_think_content(completion: str) -> tuple[str, str, int]:
    think_count = completion.count("<think>")
    think_start = completion.find("<think>")
    think_end = completion.find("</think>")

    if think_start == -1:
        return "", completion, 0

    if think_end == -1:
        reasoning = completion[think_start + len("<think>"):].strip()
        return reasoning, "", think_count

    reasoning = completion[think_start + len("<think>"):think_end].strip()
    response = completion[think_end + len("</think>"):].strip()
    return reasoning, response, think_count


def _has_expected_format(completion: str) -> bool:
    match = re.fullmatch(
        r"\s*<think>(?P<reasoning>.*?)</think>\s*(?P<response>.+?)\s*",
        completion,
        flags=re.DOTALL,
    )
    if match is None:
        return False

    reasoning = match.group("reasoning").strip()
    response = match.group("response").strip()
    return (
        bool(reasoning)
        and bool(response)
        and completion.count("<think>") == 1
        and completion.count("</think>") == 1
        and re.search(r"<\s*/?\s*answer\b[^>]*>", response, re.IGNORECASE) is None
    )


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _word_count(text: str) -> int:
    return len(text.split())


def _has_doom_loop(text: str, max_repeats: int = 3) -> bool:
    words = re.findall(r"\b[\w']+\b", text.casefold())
    repeat_count = 1

    for previous, current in zip(words, words[1:]):
        repeat_count = repeat_count + 1 if current == previous else 1
        if repeat_count > max_repeats:
            return True

    return False


# GRPO reward functions use TRL's batch callback signature.


def think_format_reward(prompts, completions, answer, **kwargs):
    return [
        1.0 if _has_expected_format(completion[0]["content"]) else 0.0
        for completion in completions
    ]


def output_format_reward(prompts, completions, answer, **kwargs):
    scores = []
    for completion, ref in zip(completions, answer):
        content = completion[0]["content"]
        if not _has_expected_format(content) or not ref:
            scores.append(OUTPUT_FORMAT_MISMATCH_REWARD)
            continue

        _, response, _ = _extract_think_content(content)
        formats_match = _is_valid_json(str(ref)) == _is_valid_json(response)
        scores.append(
            OUTPUT_FORMAT_MATCH_REWARD
            if formats_match
            else OUTPUT_FORMAT_MISMATCH_REWARD
        )
    return scores


def doom_loop_reward(prompts, completions, answer, **kwargs):
    return [
        DOOM_LOOP_PENALTY
        if _has_doom_loop(completion[0]["content"])
        else 0.0
        for completion in completions
    ]


def neuraltxt_reward(prompts, completions, answer, **kwargs):
    rm = _get_reward_model()
    scores = []

    for completion, ref in zip(completions, answer):
        content = completion[0]["content"]
        if not _has_expected_format(content):
            scores.append(0.0)
            continue

        _, response, _ = _extract_think_content(content)

        if not response or not ref:
            scores.append(0.0)
            continue

        score = float(rm.score(response=response, reference=str(ref)))

        ref_words = _word_count(str(ref))
        resp_words = _word_count(response)
        max_ok = max(ref_words * 1.5, ref_words + 20)
        if resp_words > max_ok:
            ratio = max_ok / resp_words
            score *= ratio

        scores.append(score)
    return scores


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def append_reward_signal_log(
    path: Path,
    step: int,
    logs,
    reward_weights: dict[str, float],
    references: list | None = None,
    generation_batch: int | None = None,
) -> int:
    prompts = list(logs["prompt"])
    completions = list(logs["completion"])
    advantages = list(logs["advantages"])
    rewards = {
        name: list(values)
        for name, values in logs["rewards"].items()
    }
    if references:
        sample_count = len(references)
        prompts = prompts[-sample_count:]
        completions = completions[-sample_count:]
        advantages = advantages[-sample_count:]
        rewards = {
            name: values[-sample_count:]
            for name, values in rewards.items()
        }

    row_count = min(
        len(prompts),
        len(completions),
        len(advantages),
        *(len(values) for values in rewards.values()),
    )
    if references is not None:
        row_count = min(row_count, len(references))
    if row_count == 0:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        for index in range(row_count):
            reasoning, response, _ = _extract_think_content(completions[index])
            reward_signals = {
                name: _finite_number(values[index])
                for name, values in rewards.items()
            }
            total_reward = sum(
                value * reward_weights.get(name, 1.0)
                for name, value in reward_signals.items()
                if value is not None
            )
            record = {
                "step": step,
                "generation_batch": generation_batch,
                "sample_index": index,
                "prompt": prompts[index],
                "reference": references[index] if references is not None else None,
                "completion": completions[index],
                "reasoning": reasoning,
                "response": response,
                "rewards": reward_signals,
                "reward_weights": reward_weights,
                "total_reward": total_reward,
                "advantage": _finite_number(advantages[index]),
            }
            log_file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return row_count


def eval_log_path(log_dir: Path, step: int) -> Path:
    return log_dir / f"test_{step}.jsonl"


def format_row(example):
    instruction = example["instruction"]
    inp = example.get("input", "")
    question = instruction if not inp else f"{instruction}\n\n{inp}"
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "answer": example["output"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO reasoning training with Unsloth."
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="paperbd/neuraltxt-135M-reasoning-base",
        help="HF model ID or local path (merged SFT warmup with think-tag behavior).",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="models/reasoning_grpo",
        help="Output directory for GRPO checkpoints.",
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        type=str,
        default="paperbd/paper_instructions_300K-v1",
        help="HF dataset name.",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_generations", "-g", type=int, default=4)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-6)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--num_train_rows", "-n", type=int, default=1000)
    parser.add_argument("--num_eval_rows", type=int, default=32)
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for per-evaluation-step JSONL logs.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fast_inference", action="store_true", default=False)
    return parser.parse_args()


def main():
    import sys

    args = parse_args()

    if sys.platform != "darwin":
        import transformers.utils.generic

        transformers.utils.generic._is_mlx_available = False

    from datasets import load_dataset
    from accelerate.utils import gather_object
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    log_dir = Path(args.log_dir or f"{args.output_dir}/log")

    class RewardLoggingGRPOTrainer(GRPOTrainer):
        _eval_log_step = None
        _reward_log_batch_index = 0

        def _calculate_rewards(
            self,
            inputs,
            prompts,
            completions,
            completion_ids_list,
        ):
            rewards = super()._calculate_rewards(
                inputs,
                prompts,
                completions,
                completion_ids_list,
            )
            if not self.model.training:
                references = [example.get("answer") for example in inputs]
                self._eval_log_references = gather_object(references)
            return rewards

        def _generate_and_score_completions(self, inputs):
            result = super()._generate_and_score_completions(inputs)
            if not self.model.training and self.accelerator.is_main_process:
                path = eval_log_path(log_dir, self.state.global_step)
                if self._eval_log_step != self.state.global_step:
                    path.unlink(missing_ok=True)
                    self._eval_log_step = self.state.global_step
                    self._reward_log_batch_index = 0

                reward_weights = {
                    name: float(weight)
                    for name, weight in zip(
                        self.reward_func_names,
                        self.reward_weights.tolist(),
                    )
                }
                append_reward_signal_log(
                    path,
                    self.state.global_step,
                    self._logs,
                    reward_weights,
                    getattr(self, "_eval_log_references", None),
                    self._reward_log_batch_index,
                )
            if not self.model.training:
                self._reward_log_batch_index += 1
                self._eval_log_references = []
            return result

    print(f"Loading model: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        max_lora_rank=args.lora_r,
        gpu_memory_utilization=0.8,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=args.lora_r >= 64,
    )

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.shuffle(seed=SEED)
    num_eval_rows = min(args.num_eval_rows, len(dataset))
    remaining_rows = len(dataset) - num_eval_rows
    num_train_rows = min(args.num_train_rows, remaining_rows)
    if num_train_rows == 0 or num_eval_rows == 0:
        raise ValueError("Training and evaluation datasets must both be non-empty.")

    eval_dataset = dataset.select(range(num_eval_rows))
    train_dataset = dataset.select(
        range(num_eval_rows, num_eval_rows + num_train_rows)
    )
    train_dataset = train_dataset.map(
        format_row,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        format_row,
        remove_columns=eval_dataset.column_names,
    )

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.num_generations,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_seq_length // 2,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        report_to="none",
        output_dir=args.output_dir,
        bf16=True,
        seed=SEED,
    )

    print(
        f"Model: {args.model_path}\n"
        f"Train rows: {len(train_dataset)}\n"
        f"Eval rows: {len(eval_dataset)}\n"
        f"Max steps: {args.max_steps}\n"
        f"Generations per prompt: {args.num_generations}\n"
        f"LoRA rank: {args.lora_r}\n"
    )

    trainer = RewardLoggingGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            think_format_reward,
            output_format_reward,
            doom_loop_reward,
            neuraltxt_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    model.save_pretrained(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print(f"Saved to {args.output_dir}/final")
    print(f"Evaluation logs saved to {log_dir}")


if __name__ == "__main__":
    main()
