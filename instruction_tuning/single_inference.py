import argparse

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """

parser = argparse.ArgumentParser(description="Single prompt inference.")
parser.add_argument("--model_path", "-m", type=str, required=True)
parser.add_argument("--prompt", "-p", type=str, required=True)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--mlx", action="store_true", help="Use MLX for inference.")
args = parser.parse_args()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": args.prompt},
]

if args.mlx:
    from mlx_lm import load, generate
    model, tokenizer = load(args.model_path)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_new_tokens, verbose=False)
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

print(response)
