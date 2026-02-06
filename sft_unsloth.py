from unsloth import FastLanguageModel
import torch
import argparse
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model using Unsloth.")
    parser.add_argument("--base_model_id", type=str, default="HuggingFaceTB/SmolLM-135M", help="Base model ID from Hugging Face.")
    parser.add_argument("--output_model_id", "-o", type=str, 
                        default="cpt_arxiv", 
                        help="ID for the new fine-tuned model.")
    parser.add_argument("--dataset_path", "-d", type=str, required=True, help="Path to the dataset in JSONL format.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit precision.")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Load the Model and Tokenizer with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model_id,
        max_seq_length = args.max_seq_length,
        dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = args.load_in_4bit,
    )

    # Configure LoRA with Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # Start the training process
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        dataset_num_proc = 2,
        packing = True, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            output_dir=f"models/{args.output_model_id}",
            save_total_limit=3,
            per_device_train_batch_size=16,
            num_train_epochs=50,
            save_steps=10,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
        ),
    )

    # Train
    trainer.train()

    # Save the model
    model.save_pretrained(f"models/{args.output_model_id}/final")
    tokenizer.save_pretrained(f"models/{args.output_model_id}/final")

if __name__ == "__main__":
    main()
