# Reasoning Model Evaluation: Reward System & Results

## Reward Components

The total reward signal (used in both GRPO training and offline evaluation) is a weighted sum of 5 components:

| Component | Range | Weight | Description |
|---|---|---|---|
| `think_format_reward` | {0, 1} | 1.0 | 1.0 if completion has valid `<think>...</think>` format, else 0.0 |
| `output_format_reward` | {-0.5, +0.5} | 1.0 | +0.5 if output format (JSON/text) matches reference, -0.5 otherwise |
| `doom_loop_reward` | {-1.0, 0} | 1.0 | -1.0 if 4+ consecutive repeated words, else 0.0 |
| `neuraltxt_reward` | [0, 1] | 1.0 | RAW semantic quality from NeuralTxtReward model (no length penalty) |
| `length_penalty_reward` | (≤0] | **0.5** | Penalty for excessively long answers: `raw_score * (multiplier - 1.0)` |

**Max possible total reward:** 1.0 + 0.5 + 0.0 + 1.0 + 0.0 = **2.5**
(perfect format, matching output type, no doom loop, perfect semantic score, concise answer)

The old system baked length penalty into `neuraltxt_reward` at implicit weight 1.0 — a perfect but verbose answer (10x reference length) scored 0.1. Under the new system with weight 0.5, the same answer scores 0.55. The penalty still exists but no longer destroys good verbose answers.

### Format validation (`_has_expected_format`)
A completion passes if it contains exactly one `<think>...</think>` pair, both reasoning and response sections are non-empty, and the response has no nested `<answer>` tags.

### NeuralTxtReward scoring (`score_neuraltxt`)
1. Extract only the answer (text after `<think>`) — the `<think>` reasoning is discarded
2. Score the answer against the reference using `NeuralTxtReward(backend="hf")` → the custom 22M MiniLM reward model (`paperbd/neuraltxt-reward-22M`, v7-1)
3. **Raw score returned as `neuraltxt_reward`** (weight 1.0)
4. A separate **`length_penalty_reward`** (weight 0.5) is computed as `raw_score * (length_multiplier - 1.0)`, where `length_multiplier = min(1.0, max_allowed_words / actual_words)` and `max_allowed_words = max(reference_words * 1.5, reference_words + 20)`
5. Invalid-format completions score 0.0 on both components (not sent to the reward model at all)

### Reward model characteristics
- **100-pt Spearman**: 0.718 (correlation with human judge)
- **Confound resistance**: only 12% fooled by factually-wrong-but-word-overlapping responses (vs RewardBert's 53%, Word F1's 78%)
- **Swap detection**: catches 81% antonym flips, 86% negations, 80% number swaps
- **Dynamic range**: scores span 0.14–0.90 (wide separation between good and bad responses)
- Trained with contrastive augmentation (synonym swaps, antonym swaps, negation flips, number swaps) to discriminate meaning from word overlap
- Full findings: `reward_models/REWARD_MODEL_FINDINGS.md`

## Offline Evaluation

### Scripts
| Script | Purpose |
|---|---|
| `reasoning/evaluate_base.py` | Generate rollouts from a HF model, score with all 4 reward components |
| `reasoning/eval_checkpoint.py` | Same but merges a LoRA adapter into the base model first |
| `reasoning/score_head_to_head.py` | Compare two rollout JSONL files using NeuralTxtReward batch scoring |
| `reasoning/benchmark_head_to_head.py` | Load two MLX models, generate + score head-to-head |
| `reasoning/eval_mlx_checkpoints.py` | Load MLX models, generate on test split, score with NeuralTxtReward (with SFT baseline) |

### Head-to-head scoring method
1. Both models generate completions on the same held-out test split (from `paperbd/paper_instructions_300K-v1`)
2. For reasoning models: `<think>` tags are stripped, only the answer is scored
3. For SFT models: full response is scored as-is
4. `NeuralTxtReward.batch_score(answers, references, batch_size=8)` scores all responses
5. Per-example delta = score_A - score_B; winner determined by ±0.001 threshold
6. Summary: win_rate, mean_delta, std_delta, ties

## Run 3 Results (200 samples, temp=0.0, test split)

### vs SFT baseline (`neuraltxt-v1-135M`)

| Checkpoint | NeuralTxt | Raw Score | Format Pass | Length Penalty | Resp Words | Win vs SFT |
|---|---|---|---|---|---|---|
| base (step 0) | 0.3465 | 0.5902 | 100% | 74.5% | 132.8 | 47.0% |
| 96 | 0.3075 | 0.3213 | 60.0% | 9.0% | 42.4 | 19.0% |
| 2560 | 0.3366 | 0.3437 | 69.0% | 9.5% | 42.5 | 22.0% |
| 4096 | 0.3429 | 0.3532 | 69.5% | 9.5% | 41.9 | 24.5% |
| **5120** | **0.3444** | **0.3573** | **73.0%** | 11.0% | 42.1 | 21.0% |
| **SFT** | **0.6024** | **0.6042** | 100% | 5.5% | 51.6 | — |

### vs reasoning-base (`neuraltxt-135M-reasoning-base`)

| Model | NeuralTxt | Raw Score | Format Pass | Length Penalty | Resp Words | Win Rate |
|---|---|---|---|---|---|---|
| reasoning-base | 0.3465 | 0.5902 | 100% | 74.5% | 132.8 | — |
| ckpt-5120 | 0.3444 | 0.3573 | 73.0% | 11.0% | 42.1 | **52.5%** |

**Key insight:** checkpoint-5120 and reasoning-base are tied on final NeuralTxt score (0.344 vs 0.346), but the mechanism is completely different:
- **reasoning-base**: high raw quality (0.59) but crippled by verbosity — 75% of responses hit the length penalty, dragging the mean from 0.59 → 0.35
- **checkpoint-5120**: lower raw quality (0.36) but concise — only 11% penalized, minimal penalty drag
- GRPO effectively trades verbosity for tighter, more efficient answers at similar quality

### Failure modes (checkpoint-5120, 200 samples)

| Category | Count | % |
|---|---|---|
| Valid `<think>...</think>` format | 146 | 73.0% |
| No `<think>` at all | ~11 | ~5.5% |
| Unclosed `<think>` (hits max_tokens) | ~12 | ~6.0% |
| Double `<think>` tags | ~5 | ~2.5% |
| Other format issues | ~26 | ~13.0% |

Among format-OK responses: mean raw score 0.49 (vs SFT's 0.60). Even with correct format, answer quality lags behind simple SFT.

## Inference technique for MLX models

```python
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("models/mlx/run3_checkpoint-5120")
sampler = make_sampler(temp=0.0)  # greedy

prompts = [tokenizer.encode(prompt_text) for prompt_text in prompts_list]
result = batch_generate(
    model, tokenizer,
    prompts=prompts,
    max_tokens=512,
    sampler=sampler,
    verbose=False,
)
completions = [t.strip() for t in result.texts]
```

## Merge pipeline

```bash
# 1. Merge LoRA adapter into base model
python instruction_tuning/merge_adapter.py \
    --adapter_path models/reasoning_models/run_3/checkpoint-5120 \
    --output_path models/merged/run3_checkpoint-5120

# 2. Convert to MLX
python -m mlx_lm convert \
    --hf-path models/merged/run3_checkpoint-5120 \
    --mlx-path models/mlx/run3_checkpoint-5120

# 3. Evaluate
python reasoning/eval_mlx_checkpoints.py \
    --num_samples 200 --batch_size 32 --temperature 0.0 \
    --ckpt_dirs models/mlx/run3_checkpoint-5120 \
    --sft_model paperbd/neuraltxt-v1-135M-mlx
```

## Reward Diversity at Different Temperatures

GRPO requires non-zero group std to compute meaningful advantages `(reward - group_mean) / group_std`.
If all rollouts in a group score identically, the advantage is zero and no learning occurs.

Tested with `reasoning-base` model, 30 prompts, `n_rollouts=4`, scoring with the 5-component reward system:

| Metric | Temp 0.5 | Temp 0.8 |
|---|---|---|
| Mean total reward | **0.662** | 0.355 |
| Avg group std | 0.528 | 0.547 |
| Median group std | 0.216 | 0.337 |
| Group range (avg) | 1.211 | 1.297 |
| **Zero-std groups** | **26.7%** | 33.3% |
| mean/std ratio | 1.25x | 0.65x |

**Temp 0.5 is slightly better for GRPO** — fewer zero-std groups (26.7% vs 33.3%) and higher mean reward.
The 26.7% zero-std groups are mostly cases where all 4 rollouts fail format (all score ~0).
The non-zero groups have enough signal (mean/std ~1.25x) for policy gradient learning.

### Maximum Possible Reward

| Scenario | think (1.0) | format (1.0) | doom (1.0) | neuraltxt (1.0) | length (0.5) | Total |
|---|---|---|---|---|---|---|
| Perfect, concise | 1.0 | 0.5 | 0.0 | 1.0 | 0.0 | **2.5** |
| Perfect, 3x ref length | 1.0 | 0.5 | 0.0 | 1.0 | -0.33 | 2.17 |
| Perfect, 10x ref length | 1.0 | 0.5 | 0.0 | 1.0 | -0.45 | 2.05 |
| OK answer, concise | 1.0 | 0.5 | 0.0 | 0.6 | 0.0 | 2.1 |
| Format fail | 0.0 | -0.5 | 0.0 | 0.0 | 0.0 | **-0.5** |
| Doom loop | 1.0 | 0.5 | -1.0 | 0.0 | 0.0 | **0.5** |

Under the old system (length penalty baked into neuraltxt at implicit weight 1.0), a perfect but 10x-verbose answer would score `1.0 + 0.5 + 0.0 + 0.1 = 1.6`.
The new system gives it `1.0 + 0.5 + 0.0 + 1.0 - 0.45 = 2.05` — a 28% increase, keeping verbose-but-good answers in the game.
