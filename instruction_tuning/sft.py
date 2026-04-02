from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

SEED = 3407

ALPACA_PROMPT = """Below is an instruction that describes a task, written with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = None  # set after tokenizer is loaded


def format_alpaca(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, inp, output in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(instruction, inp, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="Instruction fine-tuning with Unsloth on alpaca-format data.")
    parser.add_argument("--base_model_id", "-i", type=str, required=True,
                        help="Path to a model in models/ directory or a HF model ID.")
    parser.add_argument("--output_model_id", "-o", type=str, default="instruction_tuned",
                        help="Output subdirectory under models/.")
    parser.add_argument("--dataset", "-d", type=str, default="paperbd/paper_instructions_300K-v1",
                        help="HF dataset to train on (alpaca format).")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", "-bs", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", "-e", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--full_training", "-ft", action="store_true",
                        help="Full fine-tune (no LoRA).")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    args = parser.parse_args()

    # Load model + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=args.full_training,
    )

    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token

    if not args.full_training:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
            use_rslora=False,
            loftq_config=None,
        )

    # Dataset
    dataset = load_dataset(args.dataset, split="train")
    # Expect columns: instruction, input, output
    dataset = dataset.map(format_alpaca, batched=True, remove_columns=dataset.column_names)

    # Train/eval split
    split = dataset.train_test_split(test_size=0.02, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train: {len(train_dataset):,}  |  Eval: {len(eval_dataset):,}")

    learning_rate = 1e-5 if args.full_training else 2e-4
    max_grad_norm = 0.7 if args.full_training else 1.0

    training_args = UnslothTrainingArguments(
        output_dir=f"models/{args.output_model_id}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        learning_rate=learning_rate,
        embedding_learning_rate=learning_rate * 0.1,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=SEED,
        # sequence / packing
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=True,
        dataset_num_proc=4,
        # logging & saving
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        report_to="none",
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.add_callback(
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
    )

    trainer.train()

    # Save final
    model.save_pretrained(f"models/{args.output_model_id}/final")
    tokenizer.save_pretrained(f"models/{args.output_model_id}/final")
    print(f"Saved to models/{args.output_model_id}/final")


if __name__ == "__main__":
    main()
