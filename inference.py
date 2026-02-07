import argparse
import json
import torch
import random
import os
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
    parser = argparse.ArgumentParser(description="Run inference comparison between two models.")
    parser.add_argument("--model1", type=str, required=True, help="Path or ID of Model 1")
    parser.add_argument("--model2", type=str, required=True, help="Path or ID of Model 2")
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM-135M", help="Base model ID (needed for HF adapters)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to run")
    parser.add_argument("--prefix_len", type=int, default=100, help="Number of words for input prefix")
    parser.add_argument("--predict_len", type=int, default=50, help="Number of words to predict")
    parser.add_argument("--output_json", type=str, default="inference_results.json", help="Path to save results as JSON")
    
    args = parser.parse_args()
    
    console = Console()
    
    # Load Models
    console.rule("[bold blue]Loading Models")
    model1, tokenizer1 = load_model(args.model1, args.base_model)
    model2, tokenizer2 = load_model(args.model2, args.base_model)
    
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
    table.add_column(f"Model 1: {os.path.basename(args.model1)}", style="cyan")
    table.add_column(f"Model 2: {os.path.basename(args.model2)}", style="magenta")
    table.add_column("Ground Truth (GT)", style="green")
    
    results_list = []

    for i, line in enumerate(samples):
        data = json.loads(line)
        full_text = data.get('text', '')
        
        # Split text
        words = full_text.split()
        if len(words) < args.prefix_len + args.predict_len:
            continue # Skip if too short
            
        # Create random start point to avoid just always predicting the very beginning
        # Ensure we have enough room for prefix + prediction
        max_start = len(words) - (args.prefix_len + args.predict_len)
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = random.randint(0, max_start)
            
        prefix_words = words[start_idx : start_idx + args.prefix_len]
        gt_words = words[start_idx + args.prefix_len : start_idx + args.prefix_len + args.predict_len]
        
        prefix_text = " ".join(prefix_words)
        gt_text = " ".join(gt_words)
        
        # Run Inference
        # Note: Unsloth sometimes requires inputs to be formatted specific ways, but raw text generation usually fine
        
        with console.status(f"[bold yellow]Processing sample {i+1}/{args.num_samples}..."):
            pred1 = generate_text(model1, tokenizer1, prefix_text, max_new_tokens=len(gt_words)*2) # Allow buffer for tokens vs words
            pred2 = generate_text(model2, tokenizer2, prefix_text, max_new_tokens=len(gt_words)*2)
            
        # Truncate predictions roughly to desired word count for fair comparison, or just use raw?
        # User said "predict the next 50 words". Models generate tokens. 
        # We'll just take the generated text. 
        # Calculate Scores
        score1 = calculate_rouge_l(pred1, gt_text)
        score2 = calculate_rouge_l(pred2, gt_text)
        
        # Add to table
        table.add_row(
            prefix_text + "...",
            f"[bold]{score1:.4f}[/bold]\n{pred1}",
            f"[bold]{score2:.4f}[/bold]\n{pred2}",
            gt_text
        )

        # Collect data for JSON
        results_list.append({
            "prefix": prefix_text,
            "ground_truth": gt_text,
            "model1": {
                "name": os.path.basename(args.model1),
                "prediction": pred1,
                "rouge_l": score1
            },
            "model2": {
                "name": os.path.basename(args.model2),
                "prediction": pred2,
                "rouge_l": score2
            }
        })

    console.print(table)
    
    # Write results to JSON
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results_list, f, indent=4)
        console.print(f"\n[bold green]Results saved to {args.output_json}[/bold green]")

if __name__ == "__main__":
    main()
