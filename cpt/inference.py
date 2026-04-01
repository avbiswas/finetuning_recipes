import argparse
import json
import math
import torch
import random
import os
import glob
from rich.console import Console
from rich.progress import track
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_path, base_model_id="HuggingFaceTB/SmolLM-135M", load_in_4bit=True):
    """
    Loads a model using Unsloth (if CUDA) or Standard HF (if not).
    Ensures tokenizer is configured for left-padding (critical for batch generation).
    """
    print(f"Loading model from: {model_path}...")
    
    if torch.cuda.is_available():
        from unsloth import FastLanguageModel
        print("🚀 CUDA detected. Using Unsloth.")
        
        # Unsloth handles adapters automatically if model_path points to one
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
    else:
        print("🐢 CUDA not detected. Using standard Hugging Face.")
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        # 1. Load Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            
        # 2. Check if model_path is likely an adapter
        is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_adapter:
            print(f"Found adapter at {model_path}. Loading base model {base_model_id} first...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map=device,
                torch_dtype=torch.float16 if device == "mps" else torch.float32
            )
            model = PeftModel.from_pretrained(model, model_path)
        else:
            print(f"Loading full model from {model_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16 if device == "mps" else torch.float32
            )
            
    # CRITICAL: Set padding side to left for decoder-only batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def generate_batch(model, tokenizer, prompts, max_new_tokens=64, batch_size=4, repetition_penalty=1.2):
    """
    Generates text for a list of prompts using batch processing.
    """
    all_outputs = []
    
    # Process in chunks of batch_size
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty
            )
        
        # Decode
        # Slice to get only new tokens
        generated_tokens = outputs[:, input_length:]
        decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        all_outputs.extend([text.strip() for text in decoded_batch])
        
    return all_outputs

def compute_perplexity(model, tokenizer, prefixes, ground_truths):
    """Compute perplexity on ground truth continuations given prefixes."""
    total_loss = 0
    total_tokens = 0

    for prefix, gt in zip(prefixes, ground_truths):
        full_text = prefix + " " + gt

        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)

        prefix_len = prefix_ids.shape[1]

        labels = full_ids.clone()
        labels[0, :prefix_len] = -100  # Only compute loss on GT portion

        with torch.no_grad():
            outputs = model(input_ids=full_ids, labels=labels)

        num_gt_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * num_gt_tokens
        total_tokens += num_gt_tokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

def main():
    parser = argparse.ArgumentParser(description="Run batch inference for multiple models.")
    parser.add_argument("--models", nargs='+', required=True, help="List of model paths or IDs")
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM-135M", help="Base model ID (needed for HF adapters)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to run")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--prefix_len", type=int, default=20, help="Number of words for input prefix")
    parser.add_argument("--predict_len", type=int, default=50, help="Number of words to predict")
    parser.add_argument("--output_json", type=str, default="inference_generations.json", help="Path to save generations")
    parser.add_argument("--output_results", type=str, default=None, help="If set, run evals and save results to this path")
    
    args = parser.parse_args()
    console = Console()
    
    # 1. Prepare Dataset
    console.rule("[bold blue]Preparing Dataset")
    with open(args.dataset, 'r') as f:
        lines = f.readlines()
    
    random.seed(42)
    random.shuffle(lines)

    # Parse all valid texts (use first 70% of words to avoid reference sections)
    all_texts = []
    for line in lines:
        data = json.loads(line)
        full_text = data.get('text', '')
        words = full_text.split()
        # Only keep first 90% of each text to skip references
        words = words[:int(len(words) * 0.9)]
        if len(words) >= args.prefix_len + args.predict_len:
            all_texts.append(words)

    # Sample num_samples windows across all texts
    samples = []
    prompts = []
    ground_truths = []
    sample_id = 0

    while len(samples) < args.num_samples and all_texts:
        text_words = random.choice(all_texts)
        max_start = len(text_words) - (args.prefix_len + args.predict_len)
        start_idx = random.randint(0, max_start)

        prefix_words = text_words[start_idx : start_idx + args.prefix_len]
        gt_words = text_words[start_idx + args.prefix_len : start_idx + args.prefix_len + args.predict_len]

        prefix_text = " ".join(prefix_words)
        gt_text = " ".join(gt_words)

        samples.append({
            "id": sample_id,
            "prefix": prefix_text,
            "ground_truth": gt_text,
            "predictions": {}
        })
        prompts.append(prefix_text)
        ground_truths.append(gt_text)
        sample_id += 1

    console.print(f"Prepared {len(samples)} samples from {len(all_texts)} texts.")

    # 2. Resolve Model Paths
    expanded_model_paths = []
    for pattern in args.models:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            expanded_model_paths.extend(matches)
        else:
            expanded_model_paths.append(pattern)
    unique_paths = list(dict.fromkeys(expanded_model_paths))
    
    if not unique_paths:
        console.print("[bold red]No models found![/bold red]")
        return

    # 3. Inference Loop
    perplexities = {}
    for model_path in unique_paths:
        console.rule(f"[bold blue]Processing {model_path}")
        
        # Determine display name
        if os.path.exists(model_path):
            name = model_path.rstrip(os.sep)
            parts = name.split(os.sep)
            display_name = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else parts[-1]
        else:
            display_name = model_path
            
        try:
            # Load
            model, tokenizer = load_model(model_path, args.base_model)
            
            # Generate
            console.print(f"Generating for {len(prompts)} prompts (Batch Size: {args.batch_size})...")
            
            # Estimate max_new_tokens from predict_len words (approx 1.3 tokens per word)
            max_tokens = int(args.predict_len * 1.5)
            
            predictions = generate_batch(
                model, 
                tokenizer, 
                prompts, 
                max_new_tokens=max_tokens, 
                batch_size=args.batch_size,
                repetition_penalty=1.2
            )
            
            # Store results
            for sample, pred in zip(samples, predictions):
                sample["predictions"][display_name] = pred

            # Compute perplexity while model is loaded
            console.print(f"Computing perplexity...")
            ppl = compute_perplexity(model, tokenizer, prompts, ground_truths)
            perplexities[display_name] = ppl
            console.print(f"Perplexity: {ppl:.2f}")

            # Cleanup
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            console.print(f"[bold red]Error processing {model_path}: {e}[/bold red]")
            import traceback
            traceback.print_exc()

    # 4. Save Outputs
    console.rule("[bold blue]Saving Results")
    with open(args.output_json, 'w') as f:
        json.dump(samples, f, indent=4)
        
    console.print(f"[bold green]Generations saved to {args.output_json}[/bold green]")

    # 5. Optionally run evals
    if args.output_results:
        from evals import calculate_metrics
        console.rule("[bold blue]Running Evaluations")

        model_names = list(samples[0]["predictions"].keys())
        from rich.table import Table
        from rich import box

        table = Table(title="Evaluation Results", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Perplexity", style="blue")
        table.add_column("ROUGE-1", style="green")
        table.add_column("ROUGE-L", style="green")
        table.add_column("DistilBERT F1", style="magenta")

        final_scores = {}
        for model_name in model_names:
            preds = [s["predictions"][model_name] for s in samples if model_name in s["predictions"]]
            refs = [s["ground_truth"] for s in samples if model_name in s["predictions"]]

            console.print(f"Evaluating {model_name}...")
            scores = calculate_metrics(preds, refs)
            scores['perplexity'] = perplexities.get(model_name, None)
            final_scores[model_name] = scores

            table.add_row(
                model_name,
                f"{scores['perplexity']:.2f}" if scores['perplexity'] else "N/A",
                f"{scores['rouge1']:.4f}",
                f"{scores['rougeL']:.4f}",
                f"{scores.get('bertscore_f1', 0):.4f}",
            )

        console.print(table)

        with open(args.output_results, 'w') as f:
            json.dump(final_scores, f, indent=4)
        console.print(f"[bold green]Evaluation results saved to {args.output_results}[/bold green]")

if __name__ == "__main__":
    main()
