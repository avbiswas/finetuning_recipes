import torch

MAX_TOKENS = 500

def generate_responses(
    llm,
    inputs,
    max_new_tokens,
    eos_token_id,
    n_rollouts=4,
    top_p=0.95,
    temperature=0.5,
    do_sample=True,
):
    generated_response = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        num_return_sequences=n_rollouts,
        temperature=temperature,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )
    return generated_response


def _aggregate_token_loss(per_token, mask, loss_implementation, max_tokens):
    """Reduce a per-token quantity [B, T] to a scalar.

    Used by the policy-gradient, KL, and entropy terms alike so they share one
    normalization scheme — that's what keeps kld_weight / entropy_weight on a
    consistent scale relative to the policy loss regardless of response length.

    - grpo:    per-sequence mean then mean over responses (has a length bias).
    - dr_grpo: mean over responses of (token-sum / constant max_tokens). The
               constant is length-unbiased and acts as an lr reparametrization;
               consistent across micro-batches under gradient accumulation.
    - bnpo:    token-level mean over the whole batch (data-dependent denominator).
    """
    masked = per_token * mask
    if loss_implementation == "grpo":
        response_mask_sum = mask.sum(dim=1).clamp(min=1.0)
        return (masked.sum(dim=1) / response_mask_sum).mean()
    if loss_implementation == "dr_grpo":
        num_responses = masked.shape[0]
        return masked.sum() / (num_responses * max_tokens)
    if loss_implementation == "bnpo":
        return masked.sum() / mask.sum().clamp(min=1.0)
    raise ValueError(f"Unknown loss_implementation: {loss_implementation!r}")


def calculate_grpo_loss(
    log_probs,
    old_log_probs,
    advantages,
    full_response_mask,
    loss_implementation="dr_grpo",
    max_tokens=MAX_TOKENS,
    clip_epsilon_low=0.2,
    clip_epsilon_high=0.3,
):
    importance_sampling_ratio = torch.exp(log_probs - old_log_probs)

    unclipped = advantages * importance_sampling_ratio
    clipped = advantages * torch.clamp(
        importance_sampling_ratio, 1 - clip_epsilon_low, 1 + clip_epsilon_high
    )
    per_token_loss = -torch.min(unclipped, clipped)
    return _aggregate_token_loss(
        per_token_loss, full_response_mask, loss_implementation, max_tokens
    )


def calculate_kld_loss(
    log_probs,
    ref_log_probs,
    response_mask,
    loss_implementation="dr_grpo",
    max_tokens=MAX_TOKENS,
):
    kl = torch.exp(log_probs) * (log_probs - ref_log_probs.detach())
    kl = kl.sum(dim=-1)
    return _aggregate_token_loss(kl, response_mask, loss_implementation, max_tokens)


def calculate_entropy(
    log_probs,
    response_mask,
    loss_implementation="dr_grpo",
    max_tokens=MAX_TOKENS,
):
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return _aggregate_token_loss(
        entropy, response_mask, loss_implementation, max_tokens
    )
