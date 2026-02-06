from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datasets import load_dataset
import torch
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, TextStreamer
from peft import LoraConfig, get_peft_model, TaskType


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face LLM.")
    parser.add_argument("--base_model_id", type=str, default="HuggingFaceTB/SmolLM-135M", help="Base model ID from Hugging Face.")
    parser.add_argument("--output_model_id", "-o", type=str, 
                        default="cpt_arxiv", 
                        help="ID for the new fine-tuned model.")
    parser.add_argument("--dataset_path", "-d", type=str, required=True, help="Path to the dataset in JSONL format.")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Load the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the Hugging Face LLM
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # Configure LoRA for training
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Start the training process
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        peft_config=peft_config,
        args = SFTConfig(
            output_dir=f"models/{args.output_model_id}",
            save_total_limit=3,
            per_device_train_batch_size=16,
            num_train_epochs=50,
            save_steps=10,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_length=1024,
            learning_rate = 2e-4,
            packing = True,
            dataset_num_proc = 2,
            dataset_text_field="text",
            eos_token=tokenizer.eos_token,
            pad_token=tokenizer.pad_token
        ),
    )

    trainer.train()

    # Save the model
    trainer.model.save_pretrained(f"models/{args.output_model_id}/final")


if __name__ == "__main__":
    main()

