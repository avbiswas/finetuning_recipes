import argparse
import json
import torch
from rich.console import Console
from rich.table import Table
from rich import box
import evaluate

# Load metrics
try:
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    bertscore = evaluate.load('bertscore') 
except Exception as e:
    print(f"[bold red]Critical Error:[/bold red] Could not load metrics. Please run: uv pip install evaluate rouge_score bert_score absl-py sacrebleu")
    print(f"Error details: {e}")
    exit(1)

def calculate_metrics(predictions, references):
    """
    Calculates ROUGE, BLEU, and BERTScore for a list of predictions vs references.
    """
    results = {}
    
    # ROUGE
    rouge_res = rouge.compute(predictions=predictions, references=references)
    results['rouge1'] = rouge_res['rouge1']
    results['rouge2'] = rouge_res['rouge2']
    results['rougeL'] = rouge_res['rougeL']
    
    # BLEU
    bleu_res = bleu.compute(predictions=predictions, references=references)
    results['bleu'] = bleu_res['bleu']

    # BERTScore (optional, can be slow)
    results_bert = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
    results['bertscore_f1'] = sum(results_bert['f1']) / len(results_bert['f1'])

    return results

def main():
    parser = argparse.ArgumentParser(description="Run evaluations on generated JSON.")
    parser.add_argument("--input_json", type=str, default="inference_generations.json", help="Path to generations JSON")
    parser.add_argument("--output_results", type=str, default="eval_results.json", help="Path to save evaluation scores")
    args = parser.parse_args()
    
    console = Console()
    
    # Load Data
    with open(args.input_json, 'r') as f:
        data = json.load(f)
        
    if not data:
        console.print("[bold red]No data found in input JSON.[/bold red]")
        return

    # Identify Models
    # Look at the first sample to find model keys
    model_names = list(data[0]["predictions"].keys())
    console.print(f"Found generations for models: [cyan]{', '.join(model_names)}[/cyan]")
    
    # Organize data per model
    model_data = {name: {"preds": [], "refs": []} for name in model_names}
    
    for item in data:
        ground_truth = item["ground_truth"]
        for name in model_names:
            if name in item["predictions"]:
                pred = item["predictions"][name]
                model_data[name]["preds"].append(pred)
                model_data[name]["refs"].append(ground_truth)
    
    # Run Evaluations
    console.rule("[bold blue]Running Evaluations")
    
    final_scores = {}
    
    table = Table(title="Evaluation Results", box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("ROUGE-1", style="green")
    table.add_column("ROUGE-L", style="green")
    # table.add_column("BLEU", style="yellow")
    table.add_column("DistilBERT F1", style="magenta")
    
    for model_name in model_names:
        preds = model_data[model_name]["preds"]
        refs = model_data[model_name]["refs"]
        
        if not preds:
            continue
            
        console.print(f"Evaluating {model_name}...")
        scores = calculate_metrics(preds, refs)
        final_scores[model_name] = scores
        
        table.add_row(
            model_name,
            f"{scores['rouge1']:.4f}",
            f"{scores['rougeL']:.4f}",
            # f"{scores['bleu']:.4f}",
            f"{scores.get('bertscore_f1', 0):.4f}"
        )
        
    console.print(table)
    
    # Save results
    with open(args.output_results, 'w') as f:
        json.dump(final_scores, f, indent=4)
        
    console.print(f"\n[bold green]Evaluation results saved to {args.output_results}[/bold green]")

if __name__ == "__main__":
    main()
