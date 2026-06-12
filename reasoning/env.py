import json
import re
from collections.abc import Mapping
from typing import Any

import verifiers.v1 as vf
from neuraltxt import NeuralTxtReward

SEED = 3407
OUTPUT_FORMAT_MATCH_REWARD = 0.5
OUTPUT_FORMAT_MISMATCH_REWARD = -0.5
DOOM_LOOP_PENALTY = -1.0

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
Generate your reasoning first inside <think> and </think> tags. After </think>, generate only the requested final response.
When a structured format such as JSON is requested, the content after </think> must contain only that format, without Markdown fences or additional commentary.
    """

_reward_model: NeuralTxtReward | None = None


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


@vf.reward(weight=1.0)
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
    score = float(rm.score(response=response, reference=reference))
    max_ok = max(_word_count(reference) * 1.5, _word_count(reference) + 20)
    response_words = _word_count(response)
    if response_words > max_ok:
        score *= max_ok / response_words
    return score


def score_neuraltxt_batch(
    completions: list[str],
    references: list[str],
    reward_model: NeuralTxtReward | None = None,
) -> list[float]:
    """Score many completions with a single batched reward-model forward.

    Equivalent to mapping score_neuraltxt over the inputs, but the reward model
    is invoked once via batch_score instead of once per completion. Invalid /
    empty completions short-circuit to 0.0 without hitting the model.
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
    raw_scores = rm.batch_score(valid_responses, valid_references)

    for index, response, reference, raw in zip(
        valid_indices, valid_responses, valid_references, raw_scores
    ):
        score = float(raw)
        max_ok = max(_word_count(reference) * 1.5, _word_count(reference) + 20)
        response_words = _word_count(response)
        if response_words > max_ok:
            score *= max_ok / response_words
        scores[index] = score

    return scores


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


@vf.reward(weight=1.0)
async def output_format_reward(
    task: Mapping[str, Any],
    state: Mapping[str, Any],
) -> float:
    return score_output_format(
        str(state.get("completion") or ""),
        str(task.get("answer") or ""),
    )


@vf.reward(weight=1.0)
async def doom_loop_reward(
    task: Mapping[str, Any],
    state: Mapping[str, Any],
) -> float:
    return score_doom_loop(str(state.get("completion") or ""))


@vf.reward(weight=1.0)
async def neuraltxt_reward(
    task: Mapping[str, Any],
    state: Mapping[str, Any],
) -> float:
    return score_neuraltxt(
        str(state.get("completion") or ""),
        str(task.get("answer") or ""),
    )


def load_taskset(source_rows: list[dict] | None = None) -> vf.Taskset:
    if source_rows is not None:
        def source():
            for row in source_rows:
                yield {
                    "question": row.get("question", ""),
                    "answer": row.get("ground_truth", ""),
                    "completion": row.get("response", ""),
                }
    else:
        def source():
            yield {}

    return vf.Taskset(
        source=source,
        system_prompt=SYSTEM_PROMPT,
        rewards=[
            think_format_reward,
            output_format_reward,
            doom_loop_reward,
            neuraltxt_reward,
        ],
    )


def load_environment(source_rows: list[dict] | None = None) -> vf.Env:
    return vf.Env(taskset=load_taskset(source_rows))


async def score_examples(
    taskset: vf.Taskset,
    examples: list[dict],
) -> list[dict]:
    """Score a list of {question, response, ground_truth} dicts."""
    signals = vf.build_signals(rewards=taskset.rewards)
    results: list[dict] = []

    for row in examples:
        task = taskset.to_task({
            "question": row.get("question", ""),
            "answer": row.get("ground_truth", ""),
        })
        state = vf.State({
            "completion": row.get("response", ""),
        })

        scored = await vf.score_rollout(signals, task, state)
        results.append({
            "id": row.get("id"),
            "reward": scored.get("reward", 0.0),
            "metrics": scored.get("metrics", {}),
        })

    return results
