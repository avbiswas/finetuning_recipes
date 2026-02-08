import argparse
import json
import torch
import random
import os
import csv
import glob
from rich.console import Console
from rich.table import Table
from rich import box
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from rouge_score import rouge_scorer

def calculate_rouge_l(output, reference):
    """Calculates ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, output)
    return scores['rougeL'].fmeasure

def load_model(model_path, base_model_id="HuggingFaceTB/SmolLM-135M", max_seq_length=1024, load_in_4bit=True):
    """
    Loads a model using Unsloth (if CUDA) or Standard HF (if not).
    Handles Adapter logic for HF.
    """
    print(f"Loading model from: {model_path}...")
    
    if torch.cuda.is_available():
        from unsloth import FastLanguageModel
        print("🚀 CUDA detected. Using Unsloth.")
        
        # Unsloth handles adapters automatically if model_path points to one
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    else:
        print("🐢 CUDA not detected. Using standard Hugging Face.")
        
        # 1. Load Tokenizer (usually from base or adapter)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. Check if model_path is likely an adapter (contains adapter_config.json)
        is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

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
            
        return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """Generates text based on the prompt."""
    
    # Check if Unsloth or HF
    is_unsloth = hasattr(model, "fast_language_model") # Unsloth models often have this or we just use standard generate
    
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )
   
    # Decode
    # We only want the *newly generated* text, so we slice the tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return decoded_text.strip()

def main():
    parser = argparse.ArgumentParser(description="Run inference comparison between multiple models.")
    # Allow multiple models
    parser.add_argument("--models", nargs='+', required=True, help="List of model paths or IDs to compare (e.g. --models model1 model2 model3)")
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM-135M", help="Base model ID (needed for HF adapters)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to run")
    parser.add_argument("--prefix_len", type=int, default=20, help="Number of words for input prefix")
    parser.add_argument("--predict_len", type=int, default=50, help="Number of words to predict")
    parser.add_argument("--output_json", type=str, default="inference_results.json", help="Path to save results as JSON")
    parser.add_argument("--output_csv", type=str, default="inference_results.csv", help="Path to save results as CSV")
    
    args = parser.parse_args()
    
    console = Console()
    
    # Load Models
    console.rule("[bold blue]Loading Models")
    loaded_models = []
    
    # Expand wildcards
    expanded_model_paths = []
    for pattern in args.models:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            expanded_model_paths.extend(matches)
        else:
            # If no glob match (or not a pattern), assume it's a direct ID/path
            expanded_model_paths.append(pattern)
            
    # Remove duplicates while preserving order
    unique_paths = list(dict.fromkeys(expanded_model_paths))

    if not unique_paths:
        console.print("[bold red]No models found![/bold red]")
        return

    for model_path in unique_paths:
        console.print(f"Loading {model_path}...")
        try:
            model, tokenizer = load_model(model_path, args.base_model)
            
            # Determine display name
            if os.path.exists(model_path):
                # It's a file path
                # If it ends in 'final' or 'checkpoint-X', include the parent folder for clarity
                # e.g. 'models/my_run/final' -> 'my_run/final'
                name = model_path.rstrip(os.sep)
                parts = name.split(os.sep)
                if len(parts) >= 2:
                    # Use last two parts for uniqueness (e.g., 'cpt_arxiv_ft/checkpoint-90')
                    display_name = f"{parts[-2]}/{parts[-1]}"
                else:
                    display_name = parts[-1]
            else:
                # It's a HF ID
                display_name = model_path

            loaded_models.append({
                "name": display_name,
                "path": model_path,
                "model": model,
                "tokenizer": tokenizer,
                "scores": [] 
            })
        except Exception as e:
            console.print(f"[bold red]Failed to load {model_path}: {e}[/bold red]")

    if not loaded_models:
        console.print("[bold red]No models successfully loaded. Exiting.[/bold red]")
        return

    # Load Dataset
    console.rule("[bold blue]Loading Dataset")
    with open(args.dataset, 'r') as f:
        lines = f.readlines()
    
    # Shuffle and select samples
    random.shuffle(lines)
    samples = lines[:args.num_samples]
    
    console.print(f"Loaded {len(lines)} rows. Running inference on {args.num_samples} random samples.\n")
    
    # Create Table
    table = Table(title="Inference Comparison (ROUGE-L Score)", box=box.ROUNDED, show_lines=True)
    table.add_column("Input Prefix", style="dim", width=30)
    
    # Add column for each model
    for m in loaded_models:
        table.add_column(f"{m['name']}", style="cyan")
        
    table.add_column("Ground Truth (GT)", style="green")
    
    results_list = []
    csv_rows = []

    for i, line in enumerate(samples):
        data = json.loads(line)
        full_text = data.get('text', '')
        
        # Split text
        words = full_text.split()
        if len(words) < args.prefix_len + args.predict_len:
            continue # Skip if too short
            
        # Create random start point
        max_start = len(words) - (args.prefix_len + args.predict_len)
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = random.randint(0, max_start)
            
        prefix_words = words[start_idx : start_idx + args.prefix_len]
        gt_words = words[start_idx + args.prefix_len : start_idx + args.prefix_len + args.predict_len]
        
        prefix_text = " ".join(prefix_words)
        gt_text = " ".join(gt_words)
        
        row_cells = [prefix_text + "..."]
        json_entry = {
            "prefix": prefix_text,
            "ground_truth": gt_text,
            "predictions": {}
        }
        
        csv_row = {
            "prefix": prefix_text,
            "ground_truth": gt_text
        }
        
        with console.status(f"[bold yellow]Processing sample {i+1}/{args.num_samples}..."):
            for m in loaded_models:
                pred = generate_text(m["model"], m["tokenizer"], prefix_text, max_new_tokens=len(gt_words)*2)
                score = calculate_rouge_l(pred, gt_text)
                
                # Store score
                m["scores"].append(score)
                
                # Add to table cell
                row_cells.append(f"[bold]{score:.4f}[/bold]\n{pred}")
                
                # Add to JSON/CSV data
                json_entry["predictions"][m["name"]] = {
                    "text": pred,
                    "rouge_l": score
                }
                csv_row[f"{m['name']}_pred"] = pred
                csv_row[f"{m['name']}_score"] = score

        # Add GT
        row_cells.append(gt_text)
        table.add_row(*row_cells)
        
        results_list.append(json_entry)
        csv_rows.append(csv_row)

    console.print(table)
    
    # Calculate Average Scores
    console.rule("[bold blue]Final Average Scores")
    avg_table = Table(box=box.SIMPLE)
    avg_table.add_column("Model", style="cyan")
    avg_table.add_column("Avg ROUGE-L", style="bold green")
    
    final_scores = {}
    for m in loaded_models:
        if m["scores"]:
            avg_score = sum(m["scores"]) / len(m["scores"])
        else:
            avg_score = 0.0
        final_scores[m["name"]] = avg_score
        avg_table.add_row(m["name"], f"{avg_score:.4f}")
        
    console.print(avg_table)

    # Save Results
    if args.output_json:
        # Add summary to JSON
        output_data = {
            "summary": final_scores,
            "samples": results_list
        }
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=4)
        console.print(f"\n[bold green]JSON results saved to {args.output_json}[/bold green]")
        
    if args.output_csv and csv_rows:
        keys = csv_rows[0].keys()
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_rows)
        console.print(f"[bold green]CSV results saved to {args.output_csv}[/bold green]")

if __name__ == "__main__":
    main()
