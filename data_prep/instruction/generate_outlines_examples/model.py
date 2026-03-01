from pydantic import BaseModel
import outlines
import mlx_lm
from outlines.inputs import Chat

# AVB: I used MLX coz I am on a Mac, but you can use other stuff too, check outlines

# Ollama: https://dottxt-ai.github.io/outlines/latest/features/models/ollama/#structured-generation

# SGLang: https://dottxt-ai.github.io/outlines/latest/features/models/sglang/

# VLLM: https://dottxt-ai.github.io/outlines/latest/features/models/vllm_offline/

# Or OpenAI, Openrouter, etc: https://dottxt-ai.github.io/outlines/latest/features/models/


model_name = "mlx-community/Qwen3.5-4B-OptiQ-4bit"

model = outlines.from_mlxlm(
    *mlx_lm.load(model_name)
)

def generate_structured_output(
    messages, output_type, temp=0.2
):
    
    messages = Chat(messages)
    sampler = mlx_lm.sample_utils.make_sampler(temp=temp)

    out = model(messages, output_type, max_tokens=5000, sampler=sampler)
    out = output_type.model_validate_json(out)
    return out

def augment_data(
    out, output_type, context=None
):
    if context is not None:
        context = f"Additional context: {context}"

    if hasattr(out, "model_dump_json"):
        out = out.model_dump_json()

    messages = Chat([
        {
            "role": "system", 
            "content": f"Generate augmentation of the same data structure, without changing the data itself. Do not lose any information about the data. {context}. \n Generate text that is different that the original request"
        },
        {
            "role": "user",
            "content": out
        }
    ])

    sampler = mlx_lm.sample_utils.make_sampler(temp=0.5)
    new_out = model(messages, output_type, max_tokens=5000, sampler=sampler)
    if hasattr(output_type, "model_validate_json"):
        new_out = output_type.model_validate_json(new_out)
    return new_out 




