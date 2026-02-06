import torch
mps_device = torch.mps.is_available()

import time
from rich import print

model_name = "Qwen/Qwen3-0.6B"
if mps_device:
    from unsloth_mlx.mlx_model import FastMLXModel 
    from mlx_lm import generate 
    from unsloth_mlx.mlx_model import MLXLoraConfig

    model, tokenizer = FastMLXModel.from_pretrained(
        model_name)

    model = FastMLXModel.get_peft_model(
        model=model,
        lora_config=MLXLoraConfig(
            rank=8
        )
    )
else:
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name)
    model = FastModel.get_peft_model(
        model,
        r = 8,
        target_modules = 
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
    )

print(model)
prompt = {
    "messages": [
      {"role": "system", "content": "Answer truthfully"},
      {"role": "user", "content": "Hello"}
    ]
  }

tokenized = tokenizer.apply_chat_template(prompt['messages'], tokenize=False)

start_time = time.time()
out = generate(model, tokenizer, prompt = tokenized)
print(f"----\n {out} \n----\n")
print(f"Time taken: {(time.time() - start_time):.2f}")
