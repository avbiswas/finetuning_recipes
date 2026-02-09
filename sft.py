try:
    from unsloth import FastLanguageModel
    from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
except:
    print("cant import unsloth")
import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from transformers import TrainingArguments

SEED = 3407

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model using Unsloth (if CUDA) or standard HF (if not).")
    parser.add_argument("--base_model_id", "-i", type=str, default="HuggingFaceTB/SmolLM-135M", help="Base model ID from Hugging Face.")
    parser.add_argument("--output_model_id", "-o", type=str, 
                        default="cpt_arxiv", 
                        help="ID for the new fine-tuned model.")
    parser.add_argument("--dataset_path", "-d", type=str, required=True, help="Path to the dataset in JSONL format.")
    parser.add_argument("--test_dataset_path", "-td", type=str, required=True, help="Path to the test dataset in JSONL format for evaluation.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit precision (CUDA only).")
    parser.add_argument("--full_training", "-ft", action="store_true", help="Enable full training mode (no LoRA/PEFT).")
    parser.add_argument("--split_by_words", type=float, default=0.5, help="Word level split ratio of max_seq_length. Default 0.5.")
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    args = parser.parse_args()

    # Load the dataset
    train_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    eval_dataset = load_dataset("json", data_files=args.test_dataset_path, split="train")

    # Word-level chunking
    if args.split_by_words > 0:
        chunk_size = int(args.max_seq_length * args.split_by_words)
        overlap_ratio = 0.2
        step_size = int(chunk_size * (1 - overlap_ratio))
        print(f"🔪 Chunking dataset by words with size {chunk_size} and overlap {int(overlap_ratio*100)}% (step {step_size})...")

        def chunk_text(examples):
            all_chunks = []
            for text in examples["text"]:
                words = text.split()
                # Overlapping chunks
                for i in range(0, len(words), step_size):
                    chunk = words[i : i + chunk_size]
                    if len(chunk) > 10: # Filter tiny chunks
                        all_chunks.append(" ".join(chunk))
            return {"text": all_chunks}

        train_dataset = train_dataset.map(
            chunk_text,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            chunk_text,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
            
        print(f"✅ Datasets chunked. Train size: {len(train_dataset)} rows. Eval size: {len(eval_dataset)} rows.")

    # Common LoRA Config
    base_lora_config = dict(
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none"
    )

    # Set hyperparameters based on training mode
    if args.full_training:
        learning_rate = 1e-5
        max_grad_norm = 0.7
        neftune_noise_alpha = 5
    else:
        learning_rate = 2e-4
        max_grad_norm = 1.0
        neftune_noise_alpha = None

    print(f"🧠 Training Config - LR: {learning_rate}, Grad Norm: {max_grad_norm}, NEFTune: {neftune_noise_alpha}")

    # Common Training Config
    common_training_args = dict(
        output_dir=f"models/{args.output_model_id}",
        save_total_limit=10,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_steps=20,
        gradient_accumulation_steps = 2,
        warmup_steps = 100,
        max_length=args.max_seq_length,
        learning_rate = learning_rate,
        packing = True,
        dataset_num_proc = 2,
        dataset_text_field="text",
        seed = SEED,
        logging_steps = 1,
        max_grad_norm = max_grad_norm,
        neftune_noise_alpha = neftune_noise_alpha,
    )

    common_training_args.update(dict(
        evaluation_strategy = "steps",
        eval_steps = 20,
        load_best_model_at_end = True,
        metric_for_best_model = "loss",
        per_device_eval_batch_size = args.batch_size,
    ))

    if torch.cuda.is_available():
        print("🚀 CUDA detected. Using Unsloth for training.")

        # Load the Model and Tokenizer with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.base_model_id,
            max_seq_length = args.max_seq_length,
            load_in_4bit = True,
            full_finetuning=args.full_training
        )

        if not args.full_training:
            print("✨ Configuring LoRA with Unsloth...")
            # Configure LoRA with Unsloth
            model = FastLanguageModel.get_peft_model(
                model,
                **base_lora_config,
                use_gradient_checkpointing = "unsloth", 
                random_state = SEED,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ

            )
        else:
            print("🔥 Full training mode enabled. Skipping LoRA configuration.")
        
        peft_config = None 

        training_args = UnslothTrainingArguments(
            **common_training_args,
            embedding_learning_rate = learning_rate*0.1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
        )
        trainer = UnslothTrainer(
            model = model,
            tokenizer=tokenizer,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            peft_config=peft_config,
            args = training_args,
        )
    else:
        print("🐢 CUDA not detected. Using standard Hugging Face Transformers.")
        
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

        peft_config = None
        if not args.full_training:
            print("✨ Configuring LoRA for standard Hugging Face training...")
            peft_config = LoraConfig(
                **base_lora_config,
                task_type="CAUSAL_LM",
            )
        else:
            print("🔥 Full training mode enabled. Skipping LoRA configuration.")

        training_args = SFTConfig(
            **common_training_args,
            eos_token=tokenizer.eos_token,
            pad_token=tokenizer.pad_token,
        )

        # Start the training process
        trainer = SFTTrainer(
            model = model,
            tokenizer=tokenizer,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            peft_config=peft_config,
            args = training_args,
        )

    # Train
    trainer.train()

    # Save the model
    if torch.cuda.is_available():
        model.save_pretrained(f"models/{args.output_model_id}/final")
        tokenizer.save_pretrained(f"models/{args.output_model_id}/final")
    else:
        trainer.model.save_pretrained(f"models/{args.output_model_id}/final")
        tokenizer.save_pretrained(f"models/{args.output_model_id}/final")

if __name__ == "__main__":
    main()
