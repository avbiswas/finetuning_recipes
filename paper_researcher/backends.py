"""
Backend abstraction — hidden from the user.
Handles model loading and raw text generation for MLX and HuggingFace.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0


class Backend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    def stream(self, prompt: str, **kwargs): ...  # yields str chunks


class MLXBackend(Backend):
    def __init__(self, model_path: str):
        from mlx_lm import load
        self.model, self.tokenizer = load(model_path)

    def generate(self, prompt: str, **kwargs) -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)

        return mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
            **kwargs,
        )

    def stream(self, prompt: str, **kwargs):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        self._last_stats = {}
        response = None
        full_text = ""
        for response in stream_generate(
            self.model, self.tokenizer, prompt,
            sampler=make_sampler(temp=temperature),
            max_tokens=max_new_tokens,
        ):
            full_text += response.text
            yield response.text
        if response is not None:
            self._last_stats = {
                "tokens":      response.generation_tokens,
                "tps":         response.generation_tps,
                "peak_memory": response.peak_memory,
            }
            print(f"\n[RAW OUTPUT]\n{full_text}\n[/RAW OUTPUT]\n", flush=True)


class HFBackend(Backend):
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._torch = torch

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)

        input_ids = self.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self.model.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        return self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

    def stream(self, prompt: str, **kwargs):
        from transformers import TextIteratorStreamer
        from threading import Thread
        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        thread = Thread(target=self.model.generate, kwargs=dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        ), daemon=True)
        thread.start()
        for chunk in streamer:
            yield chunk


def load_backend(model_path: str, mlx: bool = False) -> Backend:
    if mlx:
        return MLXBackend(model_path)
    return HFBackend(model_path)
