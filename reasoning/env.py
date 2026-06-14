import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from neuraltxt import NeuralTxtReward

SEED = 3407
OUTPUT_FORMAT_MATCH_REWARD = 0.5
OUTPUT_FORMAT_MISMATCH_REWARD = -0.5
DOOM_LOOP_PENALTY = -1.0
# Flat penalty applied once a response exceeds the hard length cap, independent
# of the raw semantic score. Tunable; combines with its component weight (0.5).
LENGTH_OVERAGE_PENALTY = -1.0
# Hard length cap = max(reference_words * MULT, reference_words + FLOOR). Acts as
# a guardrail against egregious bloat, not a continuous "be shorter" objective.
LENGTH_CAP_MULTIPLIER = 2.0
LENGTH_CAP_FLOOR = 30


def _length_cap(reference: str) -> float:
    ref_words = _word_count(reference)
    return max(ref_words * LENGTH_CAP_MULTIPLIER, ref_words + LENGTH_CAP_FLOOR)

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
Generate your reasoning first inside <think> and </think> tags. After </think>, generate only the requested final response.
When a structured format such as JSON is requested, the content after </think> must contain only that format, without Markdown fences or additional commentary.
    """

_reward_model: NeuralTxtReward | None = None


@dataclass
class RewardBatch:
    total: np.ndarray
    components: dict[str, np.ndarray]


def _get_reward_model() -> NeuralTxtReward:
    global _reward_model
    if _reward_model is None:
        _reward_model = NeuralTxtReward(backend="hf")
    return _reward_model


def _extract_think_content(completion: str) -> tuple[str, str, int]:
    """Returns (reasoning, response, think_count)."""
    think_count = completion.count("<think>")
    think_start = completion.find("<think>")
    think_end = completion.find("</think>")

    if think_start == -1:
        return "", completion, 0

    if think_end == -1:
        reasoning = completion[think_start + len("<think>"):].strip()
        return reasoning, "", think_count

    reasoning = completion[think_start + len("<think>"):think_end].strip()
    response = completion[think_end + len("</think>"):].strip()
    return reasoning, response, think_count


def _has_expected_format(completion: str) -> bool:
    match = re.fullmatch(
        r"\s*<think>(?P<reasoning>.*?)</think>\s*(?P<response>.+?)\s*",
        completion,
        flags=re.DOTALL,
    )
    if match is None:
        return False

    reasoning = match.group("reasoning").strip()
    response = match.group("response").strip()
    return (
        bool(reasoning)
        and bool(response)
        and completion.count("<think>") == 1
        and completion.count("</think>") == 1
        and re.search(r"<\s*/?\s*answer\b[^>]*>", response, re.IGNORECASE) is None
    )


async def think_format_reward(
    task: Mapping[str, Any],
    state: Mapping[str, Any],
) -> float:
    completion = str(state.get("completion") or "")
    if not completion:
        return 0.0

    return 1.0 if _has_expected_format(completion) else 0.0


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _word_count(text: str) -> int:
    return len(text.split())


def _has_doom_loop(text: str, max_repeats: int = 3) -> bool:
    words = re.findall(r"\b[\w']+\b", text.casefold())
    repeat_count = 1

    for previous, current in zip(words, words[1:]):
        repeat_count = repeat_count + 1 if current == previous else 1
        if repeat_count > max_repeats:
            return True

    return False


def format_row(example: Mapping[str, Any]) -> dict[str, Any]:
    instruction = example["instruction"]
    inp = example.get("input", "")
    question = instruction if not inp else f"{instruction}\n\n{inp}"
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "answer": example["output"],
    }


def score_think_format(completion: str) -> float:
    return 1.0 if _has_expected_format(completion) else 0.0


def score_output_format(completion: str, reference: str) -> float:
    if not completion or not reference or not _has_expected_format(completion):
        return OUTPUT_FORMAT_MISMATCH_REWARD

    _, response, _ = _extract_think_content(completion)
    formats_match = _is_valid_json(reference) == _is_valid_json(response)
    return (
        OUTPUT_FORMAT_MATCH_REWARD
        if formats_match
        else OUTPUT_FORMAT_MISMATCH_REWARD
    )


def score_doom_loop(completion: str) -> float:
    return DOOM_LOOP_PENALTY if _has_doom_loop(completion) else 0.0


def score_neuraltxt(
    completion: str,
    reference: str,
    reward_model: NeuralTxtReward | None = None,
) -> float:
    if not completion or not reference or not _has_expected_format(completion):
        return 0.0

    _, response, _ = _extract_think_content(completion)
    if not response:
        return 0.0

    rm = reward_model or _get_reward_model()
    return float(rm.score(response=response, reference=reference))


def score_neuraltxt_batch(
    completions: list[str],
    references: list[str],
    reward_model: NeuralTxtReward | None = None,
) -> list[float]:
    """Score many completions with a single batched reward-model forward.

    Returns RAW semantic scores (no length penalty applied).
    For length penalty, use score_length_penalty_batch.
    """
    scores = [0.0] * len(completions)
    valid_indices: list[int] = []
    valid_responses: list[str] = []
    valid_references: list[str] = []

    for index, (completion, reference) in enumerate(zip(completions, references)):
        if not completion or not reference or not _has_expected_format(completion):
            continue
        _, response, _ = _extract_think_content(completion)
        if not response:
            continue
        valid_indices.append(index)
        valid_responses.append(response)
        valid_references.append(reference)

    if not valid_responses:
        return scores

    rm = reward_model or _get_reward_model()
    if hasattr(rm, "batch_score"):
        raw_scores = rm.batch_score(valid_responses, valid_references)
    else:
        raw_scores = [
            rm.score(response=response, reference=reference)
            for response, reference in zip(valid_responses, valid_references)
        ]

    for index, raw in zip(valid_indices, raw_scores):
        scores[index] = float(raw)

    return scores


def score_length_penalty_batch(
    completions: list[str],
    references: list[str],
) -> list[float]:
    """Flat hard-cap length penalty for each completion (values <= 0).

    Returns LENGTH_OVERAGE_PENALTY once a response exceeds the hard cap
    (see _length_cap: max(ref_words * LENGTH_CAP_MULTIPLIER, ref_words + LENGTH_CAP_FLOOR)),
    else 0.0. The penalty
    no longer depends on the raw semantic score — it is a clean cliff, not a
    continuous pull toward shorter answers.
    """
    penalties = [0.0] * len(completions)
    for i, (completion, reference) in enumerate(zip(completions, references)):
        if not completion or not reference or not _has_expected_format(completion):
            continue
        _, response, _ = _extract_think_content(completion)
        if not response:
            continue
        if _word_count(response) > _length_cap(reference):
            penalties[i] = LENGTH_OVERAGE_PENALTY
    return penalties


def score_completions(
    completions: list[str],
    references: list[str],
    reward_weights: dict[str, float] | None = None,
    reward_model: NeuralTxtReward | None = None,
) -> RewardBatch:
    raw_neuraltxt = np.array(
        score_neuraltxt_batch(
            completions,
            [str(reference or "") for reference in references],
            reward_model=reward_model,
        ),
        dtype=np.float32,
    )
    components = {
        "think_format_reward": np.array(
            [score_think_format(completion) for completion in completions],
            dtype=np.float32,
        ),
        "output_format_reward": np.array(
            [
                score_output_format(completion, str(reference or ""))
                for completion, reference in zip(completions, references)
            ],
            dtype=np.float32,
        ),
        "doom_loop_reward": np.array(
            [score_doom_loop(completion) for completion in completions],
            dtype=np.float32,
        ),
        "neuraltxt_reward": raw_neuraltxt,
        "length_penalty_reward": np.array(
            score_length_penalty_batch(
                completions,
                [str(reference or "") for reference in references],
            ),
            dtype=np.float32,
        ),
    }
    weights = reward_weights or {"length_penalty_reward": 0.5}
    total = np.zeros(len(completions), dtype=np.float32)
    for name, values in components.items():
        total += values * float(weights.get(name, 1.0))
    return RewardBatch(total=total, components=components)


def diagnose_completion(completion: str, reference: str) -> dict[str, Any]:
    """Per-example structural diagnostics (no reward model needed)."""
    _, response, _ = _extract_think_content(completion)
    has_format = _has_expected_format(completion)
    return {
        "has_expected_format": has_format,
        "schema_matches": bool(reference)
        and has_format
        and _is_valid_json(reference) == _is_valid_json(response),
        "has_doom_loop": _has_doom_loop(completion),
        "reference_is_json": _is_valid_json(reference),
        "response_is_json": _is_valid_json(response),
        "reference_words": _word_count(reference),
        "response_words": _word_count(response),
    }


def score_rollouts(
    rows: list[dict],
    reward_model: NeuralTxtReward | None = None,
) -> list[dict]:
    """Score offline rollout rows with the SAME reward used in training.

    Each row must have "completion" and "reference". Returns the rows enriched
    with per-component "rewards", structural "diagnostics", and "total_reward" —
    all derived from score_completions so offline evals never diverge from
    training reward.
    """
    completions = [row["completion"] for row in rows]
    references = [str(row.get("reference") or "") for row in rows]
    reward_batch = score_completions(completions, references, reward_model=reward_model)

    results = []
    for index, row in enumerate(rows):
        reasoning, response, _ = _extract_think_content(completions[index])
        rewards = {
            name: float(values[index])
            for name, values in reward_batch.components.items()
        }
        results.append(
            {
                **row,
                "reasoning": reasoning,
                "response": response,
                "diagnostics": diagnose_completion(completions[index], references[index]),
                "rewards": rewards,
                "total_reward": float(reward_batch.total[index]),
            }
        )
    return results


def summarize(results: list[dict]) -> dict[str, Any]:
    """Aggregate score_rollouts output into summary statistics."""

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    totals = [row["total_reward"] for row in results]
    total_mean = mean(totals)
    reward_names = list(results[0]["rewards"]) if results else []

    return {
        "num_rollouts": len(results),
        "mean_total_reward": total_mean,
        "std_total_reward": (
            (mean([(value - total_mean) ** 2 for value in totals])) ** 0.5
            if totals
            else 0.0
        ),
        "min_total_reward": min(totals) if totals else 0.0,
        "max_total_reward": max(totals) if totals else 0.0,
        "mean_rewards": {
            name: mean([row["rewards"][name] for row in results])
            for name in reward_names
        },
        "think_format_pass_rate": mean(
            [float(row["diagnostics"]["has_expected_format"]) for row in results]
        ),
        "schema_match_rate": mean(
            [float(row["diagnostics"]["schema_matches"]) for row in results]
        ),
        "doom_loop_rate": mean(
            [float(row["diagnostics"]["has_doom_loop"]) for row in results]
        ),
        "length_penalty_rate": mean(
            [float(row["rewards"].get("length_penalty_reward", 0.0) < 0.0) for row in results]
        ),
    }


def trl_reward_functions():
    """Return synchronous reward adapters for TRL's GRPO trainer."""

    def think_format_reward(prompts, completions, answer, **kwargs):
        del prompts, answer, kwargs
        return [
            score_think_format(completion[0]["content"])
            for completion in completions
        ]

    def output_format_reward(prompts, completions, answer, **kwargs):
        del prompts, kwargs
        return [
            score_output_format(completion[0]["content"], str(reference or ""))
            for completion, reference in zip(completions, answer)
        ]

    def doom_loop_reward(prompts, completions, answer, **kwargs):
        del prompts, answer, kwargs
        return [
            score_doom_loop(completion[0]["content"])
            for completion in completions
        ]

    def neuraltxt_reward(prompts, completions, answer, **kwargs):
        del prompts, kwargs
        return score_neuraltxt_batch(
            [completion[0]["content"] for completion in completions],
            [str(reference or "") for reference in answer],
        )

    return [
        think_format_reward,
        output_format_reward,
        doom_loop_reward,
        neuraltxt_reward,
    ]


async def output_format_reward(
    task: Mapping[str, Any],
    state: Mapping[str, Any],
) -> float:
    return score_output_format(
        str(state.get("completion") or ""),
        str(task.get("answer") or ""),
    )


async def doom_loop_reward(
    task: Mapping[str, Any],
    state: Mapping[str, Any],
) -> float:
    return score_doom_loop(str(state.get("completion") or ""))


async def neuraltxt_reward(
    task: Mapping[str, Any],
    state: Mapping[str, Any],
) -> float:
    return score_neuraltxt(
        str(state.get("completion") or ""),
        str(task.get("answer") or ""),
    )
