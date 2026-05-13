"""
Preference tuning with Unsloth + TRL.
Supports DPO and ORPO.

Usage:
    uv run python preference_optimization/train_preference.py \
        --method dpo \
        --output_model_id preference_tuned

    uv run python preference_optimization/train_preference.py \
        --method orpo \
        --output_model_id preference_orpo
"""

from __future__ import annotations

import argparse

try:
    from preference_optimization.trl_compat import patch_trl_optional_dependency_checks
except ModuleNotFoundError:
    from trl_compat import patch_trl_optional_dependency_checks

from datasets import load_dataset
from transformers import EarlyStoppingCallback
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

SEED = 3407


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preference tuning (DPO / ORPO) with Unsloth.")
    p.add_argument("--base_model_id", type=str, default="paperbd/smollm_135M_neuraltxt_v1")
    p.add_argument("--output_model_id", "-o", type=str, default="preference_tuned")
    p.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="paperbd/paper_preference_150K-v1",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["dpo", "orpo"],
        default="dpo",
        help="Preference optimization method.",
    )
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--batch_size", "-bs", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", "-e", type=int, default=3)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--learning_rate", "-lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=0.1, help="DPO/ORPO beta hyperparameter.")
    p.add_argument("--max_prompt_length", type=int, default=1536)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Method: {args.method.upper()}")
    print(f"Base model: {args.base_model_id}")
    print(f"Dataset: {args.dataset}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=args.lora_r >= 64,
        loftq_config=None,
    )

    dataset = load_dataset(args.dataset, split="train")
    split = dataset.train_test_split(test_size=0.02, seed=SEED)
    train_dataset = split["train"]
    val_dataset = split["test"]

    output_dir = f"models/{args.output_model_id}"

    if args.method == "dpo":
        PatchDPOTrainer()
        patch_trl_optional_dependency_checks()
        from trl import DPOTrainer, DPOConfig

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=DPOConfig(
                output_dir=output_dir,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                warmup_ratio=0.03,
                warmup_steps=5,
                num_train_epochs=args.epochs,
                learning_rate=args.learning_rate,
                logging_steps=10,
                dataloader_num_workers=8,
                optim="adamw_8bit",
                weight_decay=0.001,
                lr_scheduler_type="linear",
                report_to="none",
                seed=SEED,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                save_strategy="steps",
                save_steps=50,
                save_total_limit=3,
                eval_strategy="steps",
                eval_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
            ),
            beta=args.beta,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            max_prompt_length=args.max_prompt_length,
        )
    else:
        # ORPO
        patch_trl_optional_dependency_checks()
        from trl import ORPOTrainer, ORPOConfig

        trainer = ORPOTrainer(
            model=model,
            args=ORPOConfig(
                output_dir=output_dir,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                warmup_ratio=0.03,
                warmup_steps=5,
                num_train_epochs=args.epochs,
                learning_rate=args.learning_rate,
                logging_steps=10,
                dataloader_num_workers=8,
                optim="adamw_8bit",
                weight_decay=0.001,
                lr_scheduler_type="linear",
                report_to="none",
                seed=SEED,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                save_strategy="steps",
                save_steps=50,
                save_total_limit=3,
                eval_strategy="steps",
                eval_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
            ),
            beta=args.beta,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            max_prompt_length=args.max_prompt_length,
        )

    trainer.add_callback(
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
    )

    trainer.train()

    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()
