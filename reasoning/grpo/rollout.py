from dataclasses import dataclass

import numpy as np
import torch

from grpo_utils import generate_responses


@dataclass
class RolloutBatch:
    full_responses: torch.Tensor
    response_masks: torch.Tensor
    old_log_probs: torch.Tensor
    completion_texts: list[str]
    references: list[str]
    items: list[dict]
    token_counts: np.ndarray
    num_prompts: int
    group_size: int
    pad_token_id: int

    def to_experiences(self, advantages: torch.Tensor):
        padded_responses = (self.full_responses != self.pad_token_id).int()
        start_of_response = torch.argmax(padded_responses, dim=1)
        end_of_response = padded_responses.shape[1] - torch.argmax(
            torch.flip(padded_responses, dims=[1]), dim=1
        )
        return [
            (
                self.full_responses[i][start_of_response[i] : end_of_response[i]],
                self.response_masks[i][start_of_response[i] : end_of_response[i] - 1],
                self.old_log_probs[i][start_of_response[i] : end_of_response[i] - 1],
                advantages[i],
            )
            for i in range(len(self.full_responses))
        ]


def calculate_log_probs(model, input_ids, attention_masks):
    logits = model(input_ids=input_ids, attention_mask=attention_masks).logits
    full_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    token_log_probs = torch.gather(
        full_log_probs,
        dim=2,
        index=input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    return token_log_probs, full_log_probs


def calculate_token_log_probs(model, input_ids, attention_masks, chunk_size):
    """Per-token log-probs only, computed in batch-chunks to bound peak memory.

    Equivalent to calculate_log_probs(...)[0], but never materializes the full
    [B, T, vocab] log-softmax over the whole batch at once — that tensor is ~23GB
    at rollout_batch_size*n_rollouts and OOMs. Chunking over the batch is exact
    (each sequence's log-probs are independent given its own attention mask).
    """
    chunks = []
    for start in range(0, input_ids.shape[0], max(chunk_size, 1)):
        ids = input_ids[start : start + max(chunk_size, 1)]
        mask = attention_masks[start : start + max(chunk_size, 1)]
        logits = model(input_ids=ids, attention_mask=mask).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=2, index=ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        chunks.append(token_log_probs)
        del logits, log_probs
    return torch.cat(chunks, dim=0)


def collect_rollouts(
    generation_model,
    tokenizer,
    batch,
    max_new_tokens,
    n_rollouts,
    top_p,
    temperature,
    logprob_chunk_size,
):
    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    num_prompts, input_size = inputs["input_ids"].shape
    items = batch["item"]

    # Merge the LoRA adapter into the base weights for the decode loop so each
    # of the (hundreds of) generation steps runs at plain-base speed instead of
    # recomputing base + A@B every layer. Unmerge afterwards so training (and the
    # old_log_probs forward below) sees the adapter as separate parameters.
    generation_model.merge_adapter()
    try:
        full_responses = generate_responses(
            generation_model,
            inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            n_rollouts=n_rollouts,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
        )
    finally:
        generation_model.unmerge_adapter()

    responses = full_responses[:, input_size:]
    attention_masks = torch.cat(
        [
            torch.repeat_interleave(batch["attention_mask"], n_rollouts, dim=0),
            (responses != tokenizer.pad_token_id).to(torch.int64),
        ],
        dim=1,
    )
    # old_log_probs via a forward over the (unmerged, live-adapter) policy — the
    # SAME code path and weights as the training forward, so the importance ratio
    # is exactly 1.0 at step 0. (Deriving these from generation logits instead
    # diverges ~15% per token in bf16 because merged-gen != unmerged-forward.)
    # Chunked over the batch: the full B*n_rollouts forward materializes a ~23GB
    # log-softmax and OOMs; chunking bounds peak memory (exact, sequences are
    # independent).
    old_log_probs = calculate_token_log_probs(
        generation_model,
        full_responses,
        attention_masks,
        chunk_size=logprob_chunk_size,
    )

    completion_texts = tokenizer.batch_decode(responses, skip_special_tokens=True)
    token_counts = (
        (responses != tokenizer.eos_token_id)
        .to(torch.int32)
        .sum(axis=-1)
        .cpu()
        .numpy()
    )

    padded_responses = (full_responses != tokenizer.pad_token_id).int()
    end_of_response = padded_responses.shape[1] - torch.argmax(
        torch.flip(padded_responses, dims=[1]), dim=1
    )
    response_masks = torch.zeros_like(full_responses[:, 1:])
    for index in range(len(full_responses)):
        response_masks[index, input_size - 1 : end_of_response[index] - 1] = 1

    return RolloutBatch(
        full_responses=full_responses.cpu(),
        response_masks=response_masks.cpu(),
        old_log_probs=old_log_probs.cpu(),
        completion_texts=completion_texts,
        references=[item["answer"] for item in np.repeat(items, n_rollouts)],
        items=np.repeat(items, n_rollouts).tolist(),
        token_counts=token_counts,
        num_prompts=num_prompts,
        group_size=n_rollouts,
        pad_token_id=tokenizer.pad_token_id,
    )
