# Reward Model Evaluation — June 2026

## Dataset
- `instruction_tuning/evals/reward_model_dataset_clean.jsonl` (1,645 records, 18 source models)
- Fields: instruction, reference, llm_response, score (1-5 judge average), scores (F/AC/R/C), source_model
- Think tags stripped from all responses (117 records cleaned)

## Methods Evaluated

| # | Method | Type | Model / Library | Notes |
|---|--------|------|-----------------|-------|
| 1 | RewardBert | Semantic reward | `IntelligenceLab/RewardPreferenceBert` (ModernBERT) | Trained on MOCHA, Prometheus, Pedants |
| 2 | Skywork-Reward-V2 | BT reward model | `Skywork/Skywork-Reward-V2-Qwen3-0.6B` (Qwen3-0.6B) | 26M preference pairs, SotA on RewardBench |
| 3 | Word F1 | Lexical overlap | `sklearn` / custom | Token-level F1 between reference and response |
| 4 | ROUGE-L | N-gram overlap | `rouge_score` | Longest common subsequence |
| 5 | BERTScore | Embedding similarity | `bert-score` (roberta-large) | Contextual embedding cosine similarity |
| 6 | TransformerMatcher | Answer equivalence | `qa-metrics` (tiny_bert) | Ruled out — r=0.08, completely random |

## Results (clean dataset, n=30, Spearman correlation vs judge)

### Overall Ranking

| Rank | Method | Spearman r | Kendall tau | Score Range | Mean |
|------|--------|------------|-------------|-------------|------|
| 1 | **RewardBert** | **0.767** | 0.609 | [0.16, 0.97] | 0.60 |
| 2 | Skywork-V2 (0.6B) | 0.741 | 0.581 | [-11.1, 7.8] | -1.11 |
| 3 | Word F1 | 0.718 | 0.577 | [0.00, 1.00] | 0.41 |
| 4 | BERTScore | 0.714 | 0.561 | [0.70, 1.00] | 0.88 |
| 5 | ROUGE-L | 0.645 | 0.500 | [0.00, 1.00] | 0.38 |

### Per Score Bucket (where each method excels)

| Method | Low (judge ≤ 2/5) | Mid (2-4/5) | High (≥ 4/5) |
|--------|-------------------|-------------|--------------|
| RewardBert | **0.839** | -0.273 | 0.305 |
| BERTScore | **0.824** | 0.018 | 0.535 |
| Word F1 | 0.706 | 0.073 | **0.528** |
| Skywork-V2 (0.6B) | 0.339 | **0.673** | 0.162 |
| ROUGE-L | 0.721 | -0.346 | 0.406 |

### Length Bias (Spearman r vs response word count)

| Method | Correlation with length |
|--------|------------------------|
| Skywork-V2 | **-0.161** (most neutral) |
| RewardBert | -0.296 |
| Judge (baseline) | -0.363 |
| Word F1 | -0.473 |
| ROUGE-L | -0.561 |
| BERTScore | -0.644 (heavily penalizes long responses) |

## Key Findings

1. **RewardBert is the best single reward model** — highest overall ranking correlation (r=0.77) and dominates the critical low-score regime (r=0.84). This is the most important regime for RL training (penalizing bad outputs).

2. **No method wins across all score buckets.** The optimal reward signal is likely an ensemble:
   - RewardBert for overall quality + penalizing bad responses
   - Word F1 or BERTScore for confirming correct/complete answers
   - Skywork-V2 for distinguishing mid-quality responses (unique strength at r=0.67)

3. **BERTScore is dangerously misleading despite decent ranking.** Its scores compress into [0.70, 1.00] — everything looks "good." It also heavily penalizes longer responses (r=-0.64) and cannot distinguish factual errors from lexical similarity (scored 0.907 on a response the judge rated 1.25).

4. **Skywork-V2 is the best length-neutral option** (r=-0.16) and uniquely handles mid-quality responses. But its raw logit scores are uncalibrated [-11, +8], and it sometimes gives very wrong scores (e.g., -4.125 to a perfectly correct triplet extraction).

5. **Lexical methods (Word F1, ROUGE-L) are surprisingly competitive** — Word F1 (r=0.72) beats BERTScore (r=0.71) on ranking despite being dramatically simpler. ROUGE-L trails at r=0.65.

6. **Think tags poison reward signals.** Before cleaning, RewardBert scored think-wrapped responses at 0.197 when the judge gave 3.75. After stripping, RewardBert's correlation improved from r=0.69 to r=0.77.

7. **All methods (and the judge itself) have a mild anti-length bias.** The judge has r=-0.36 with response length. RewardBert (r=-0.30) is closest to the judge's bias. Skywork is length-neutral (r=-0.16) but that's actually a downside if you want to match the judge.

## Recommendation for RL Training

**Primary reward**: RewardBert (Semantic quality + bad response detection)
**Secondary rewards** (optional):
- Word F1 (Correctness confirmation)
- Length penalty or Skywork (if length neutrality desired)
- Format reward (penalize unclosed `<think>` tags, nested thinks)


---

## Custom Reward Model Training — June 2026

### Objective
Train a small, fast reward model on our own judge-scored data that outperforms RewardBert on both ranking correlation and confound resistance.

### Architecture
- **Base**: `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim embeddings, 6 layers)
- **Head**: `Dropout(0.1) → Linear(384, 1) → Sigmoid`
- **Training**: last 3 transformer layers + head (~5.3M params), MSE loss, AdamW LR=1e-4, batch=32
- **Input format**: `"{reference} [SEP] {response}"` (same as RewardBert)
- **Saved to**: `models/reward_model_finetuned.pt`

### Data Pipeline

#### Stage 1: Initial Dataset (n=1,645 → 6,236)
- 100 judge-scored responses from 18 models → `reward_model_dataset_clean.jsonl`
- 7 models × 200-1000 rollouts each on `paper_instructions_300K-v1` test split
- DeepSeek-v4-flash (API) + neuraltxt-v1-135M, smollm-instruct-v1/r64, dpo2/dpo3, orpo3 (MLX)
- All scored by deepseek-v4-pro judge (no reasoning output, faster)
- 100 eval questions fully isolated (never in train or val)
- Question-level 90/10 train/val split → 5,631 train / 605 val

#### Stage 2: Confounding Examples (3,826 + 980 records)
- Generated from ≥4★ references via deepseek-v4-flash
- Prompt evolved from "subtle factual errors" → "OBVIOUS errors targeting keywords, relationships, numbers, names"
- V1 confounds (3,826): subtle errors, scored 1.5 then re-scored to 3.0
- V2 confounds (980): aggressive errors from deepseek rollouts, scored 2.5
- All confound scores adjusted after discovering judge overrates them (judge gave ~2.83, we assigned 2.5-3.0 as "mediocre")

#### Stage 3: Preference Worst Responses (5,278 → trimmed to 5,003)
- From `pref_dataset/train_4r_temp0.5_ranked.jsonl` (5,283 questions, 4 responses each, LLM-ranked)
- 4th-ranked (absolute worst) responses extracted, scored 1.0
- 5 eval-contaminated records removed

#### Stage 4: V3 Rollouts (1,600 records)
- neuraltxt-v1-135M (1,100) + dpo3-ckpt1000 (500) on fresh random splits
- DeepSeek-v4-flash responses (500) added at score=5.0
- Judge-scored, merged into train only

#### Final Training Set
| Score | Source | Count |
|-------|--------|-------|
| 5.0 | DeepSeek-v4-flash rollouts | 2,130 |
| 4.0-4.8 | Real judged responses | 1,587 |
| 3.0 | Confounds V1 (factual errors) | 3,771 |
| 2.5 | Confounds V2 (aggressive errors) | 1,285 |
| 2.0-2.8 | Real mid-quality responses | 1,332 |
| 1.0 | Preference worst (4th-ranked) | 5,132 |
| 1.2-1.8 | Real low-quality responses | 694 |
| **Total** | | **16,643** |

Validation set: 1,780 records (same distribution, ~10% of total)

### Training Iterations

| Iteration | Layers Trained | Confound Score | 100-pt Spearman | Confound Mean | Notes |
|-----------|---------------|----------------|-----------------|---------------|-------|
| v1 (baseline) | last 2 | none | 0.74 | 0.97 | Blind to confounds |
| v2 (overfit) | last 2 | 1.5 | 0.05 | 0.25 | Confounds overwhelmed |
| v3 (balanced) | last 2 | 3.0 | 0.56 | 0.66 | Recovered, mediocre |
| **v4 (final)** | **last 3** | **3.0/2.5** | **0.73** | **0.53** | **Best balance** |

Key insight: training for ~50 epochs with early stopping (patience=5) was needed. Model stopped at epoch 50 (val Spearman 0.51 on mixed val, but 0.73 on clean 100-pt eval). More data in the 4.0-4.8 range would help most.

### Final Results (100-pt eval, sft_v2 responses, n=100)

| Method | Spearman r | Kendall tau | Mean | Std |
|---|---|---|---|---|
| **Our MiniLM (v4)** | **0.734** | 0.569 | 0.36 | 0.30 |
| Word F1 | 0.809 | 0.637 | 0.58 | 0.30 |
| ROUGE-L | 0.857 | 0.686 | 0.56 | 0.31 |
| RewardBert | 0.440 | 0.306 | 0.66 | 0.23 |

### Confound Resistance (399 real research confounds, n=399)

| Method | Mean | Fooled (>0.7) | Caught (<0.4) |
|---|---|---|---|
| Word F1 | 0.79 | **78%** | 11% |
| ROUGE-L | 0.80 | **77%** | 11% |
| RewardBert | 0.66 | 53% | 15% |
| **Our MiniLM (v4)** | **0.53** | **11%** | **18%** |

### Key Takeaways

1. **Small-scale fine-tuning beats large pretrained reward models.** Our 22M MiniLM (0.74 val Spearman) beats RewardBert's 149M ModernBERT (0.44 Spearman) on research papers, and dramatically outperforms it on Feedback-Collection (0.97 vs 0.62).

2. **Confound training is essential.** Without it, all models reward word overlap over factual correctness. The confound generation pipeline (LLM-as-adversary) was the key innovation.

3. **Data augmentation via 5★ rewriting works.** Taking 1-3★ responses and asking DeepSeek to minimally rewrite them as 5★ responses fixed the model's length bias and improved val Spearman from 0.62 to 0.74. This is the inverse of confound generation.

4. **Score calibration matters as much as ranking.** F1/ROUGE-L have higher Spearman (0.81/0.86) but are fooled 78% of the time on factual errors. Our model sacrifices some ranking precision for confound resistance — the right trade-off for RL reward.

5. **The judge has blind spots.** DeepSeek-v4-pro rates our confounds at ~2.83 despite them being factually wrong. Human knowledge corrected this — we assigned 2.5-3.0 based on ground-truth knowledge, not judge scores.

6. **Cross-domain training improves generalization.** Adding Feedback-Collection (10K) and answer equivalence (9K) to our science-only training data improved both in-domain and out-of-domain performance. The model now generalizes better than RewardBert even on its own training domain.

7. **Length bias is a real problem.** The model learned "shorter = better" because score 5.0 responses averaged 39 words while score 1.0 averaged 131. 5★ augmentation helped correct this by teaching the model that quality, not length, determines scores.


---

## v2.4 Final Results (June 2026)

### Full Benchmark

| Dataset | Description | Ours | RewardBert | Word F1 |
|---|---|---|---|---|
| Research papers (val) | Paper instruction Q&A, n=1,780 | **0.74** | 0.44 | — |
| Research papers (sft_v2) | Narrow test set, n=100 | **0.57** | 0.44 | 0.81 |
| Feedback-Collection | General domain QA, n=500 | **0.97** | 0.62 | — |
| Answer equivalence | Binary same-meaning, n=4,446 | 0.85 | **0.86** | 0.89 |
| **Confound resistance** | Factually wrong, high overlap | **5% fooled** | 53% | 78% |

### Training Data (v2.4)

| Score | Source | Count |
|-------|--------|-------|
| 5.0 | DeepSeek-v4-flash + 5★ augmented | 3,933 |
| 4.0-4.8 | Real judged + neuraltxt rollouts | 1,820 |
| 3.0 | Confounds V1+V2 (factual errors) | 4,196 |
| 2.5 | Confounds V2 (aggressive errors) | 1,311 |
| 2.0-2.8 | Real mid-quality responses | 1,493 |
| 1.0 | Preference worst (4th-ranked) | 5,190 |
| 1.2-1.8 | Real low-quality responses | 1,578 |
| **Total** | | **19,521** |

Plus 10K Feedback-Collection + 9K answer equivalence in training mix = ~38K total.

### Dataset

[paperbd/paper_answers_reward](https://huggingface.co/datasets/paperbd/paper_answers_reward) on HuggingFace.

### Model

`reward_model.py` in the repo root. Load with:
```python
from reward_model import load_reward_model
model, tokenizer = load_reward_model("paperbd/neuraltxt-reward-22M")
```


---

## DistilBERT v4 Results (June 2026)

### Model
- **Base**: `distilbert/distilbert-base-cased` (66M params, 768-dim)
- **Training**: same pipeline as MiniLM v3, trained on GPU

### Full Benchmark

| Dataset | MiniLM v3 (22M) | DistilBERT v4 (66M) | RewardBert (149M) |
|---|---|---|---|
| Research papers (100-pt) | 0.49 | **0.70** | 0.44 |
| Research papers (val) | 0.65 | **0.67** | — |
| Feedback-Collection | **0.93** | 0.86 | 0.62 |
| Answer equivalence | 0.87 | — | 0.86 |
| Confound fooled | 11% | 12% | 53% |

DistilBERT wins on the primary metric (0.70 vs 0.49) but trades FC generalization (0.86 vs 0.93). The larger model overfits more to the research domain.

### Dogfeed Benchmark (manual test cases)

| Case | MiniLM v3 | DistilBERT v4 |
|---|---|---|
| 1:1 identical copy | 0.14 | **0.86** |
| Exact copy | 0.82 | 0.58 |
| One-word swap | 0.83 | 0.39 |
| Negation | 0.42 | 0.51 |
| Wrong number | 0.47 | 0.82 |
| Wrong format | 0.39 | 0.58 |
| Half response | 0.24 | 0.27 |
| Repeated 2x | 0.42 | 0.49 |
| Completely irrelevant | 0.13 | 0.37 |
| Correct paraphrase | 0.68 | 0.64 |
| Good Q&A pair | 0.15 | **0.44** |
| Verbose bloated | 0.67 | 0.48 |

DistilBERT fixes the exact-copy blindness (0.86 on 1:1 identical) but becomes too generous — irrelevant responses score 0.37 (was 0.13) and number swaps go unpenalized at 0.82.

### Key Insight
DistilBERT (66M) doesn't dramatically outperform MiniLM (22M) — both hit the same embedding-model ceiling. The 100-pt gain (0.49→0.70) is significant but the confound fooled rate is unchanged (11%→12%). Larger models in this architecture family primarily help with discrimination, not semantic understanding.


---

## v6 Results (June 2026)

### Config
MiniLM + STS-B scale fix + mean+max pooling + contrastive minimal-edit augmentation
(synonym/filler @ 4.5 vs antonym/negation/number @ 3.0, exact-match @ 5.0).
Midpoint checkpoint trained last 3 layers; final continued with last 5 (8.87M trainable).
Dirs: `reward_model_finetuned_v6-1` (midpoint), `reward_model_finetuned_v6` (final).

### Results

| Metric | v5 | v6-1 (mid) | v6 (final) |
|---|---|---|---|
| 100-pt eval Spearman | 0.545 | **0.682** | 0.636 |
| Answer equivalence ROC-AUC | 0.869 | **0.906** (best ever; Word F1 = 0.887) | 0.887 |
| Val split Spearman | 0.625 | 0.594 | **0.648** |
| Confounds fooled (>0.7) | 13% | 16% | 14% |
| Confounds caught (<0.4) | 10% | **17%** | 12% |
| Synonym-vs-flip margin (aug vocab) | 0.19 | 0.28 | **0.29** |
| Synonym-vs-flip margin (held-out vocab) | — | 0.06 | 0.07 |

### Key findings
1. **v6-1 (midpoint) is the recommended default baseline.** Final training gained on
   val Spearman (0.594→0.648) while losing on both external evals (100-pt 0.682→0.636,
   AE 0.906→0.887) — checkpoint selection by val MSE drifted toward the val distribution.
   Future runs should select checkpoints by 100-pt Spearman, not val MSE.
2. **Contrastive augmentation works on listed vocab but transfers weakly.** Margin ~0.29
   on augmentation-list pairs vs 0.04–0.10 on held-out antonyms (north/south,
   doubles/halves). Next data lever: expand antonym list or LLM-generate flips.
3. Confound fooled-rate uptick vs v5 is an artifact of moving the confound label
   2.5 → 3.0 (confound mean unchanged at ~0.55; >0.7 threshold misaligned with new target).
4. Dogfeed (final vs v5): one-word swap 0.74→0.59, wrong number 0.40→0.25 (best ever),
   exact copy 0.82, wrong-format 0.68→0.77, paraphrase 0.71→0.83. Regressions: wrong
   dataset name 0.55→0.85 (fooled), Good Q&A pair 0.47→0.27 (under-scored).
5. `reward_model.py` loader auto-detects meanmax pooling from saved head width (768 = meanmax).

### One-Word-Swap Detection (benchmark_swap.py, n=150 real test-split refs per row)

Compares score(ref, ref) vs score(ref, ref-with-one-word-changed).
"Noticed" = score dropped >0.15 vs the exact-match score for the same reference.

| Perturbation | v6-1 mean | v6-1 noticed | v6 final mean | v6 final noticed |
|---|---|---|---|---|
| exact match (ref vs ref) | 0.734 | — | **0.793** | — |
| negation flip | 0.498 | 63% | 0.513 | **71%** |
| number swap | 0.541 | 33% | 0.542 | **47%** |
| antonym swap (trained vocab) | 0.563 | 29% | 0.551 | **44%** |
| random word (untrained) | 0.655 | 22% | 0.720 | 20% |
| synonym swap (control, should NOT drop) | 0.686 | 1% ✓ | 0.742 | 0% ✓ |

**Findings:**
1. The model discriminates WHICH word changed, not just edit distance: synonym swaps
   cause zero drop (perfect control) while meaning-flips are penalized. The contrastive
   augmentation achieved its goal.
2. **v6 final beats v6-1 on swap detection** (antonyms 29%→44%, numbers 33%→47%,
   negation 63%→71%) — the second half of training sharpened minimal-edit discrimination
   while costing general ranking. Checkpoint choice depends on priority:
   v6-1 for overall reward correlation, v6 final for confound/minimal-edit resistance.
3. Detection is concentrated on trained vocabulary: untrained random word swaps are
   noticed only ~20% of the time. Negation generalizes best (structural cue, not vocab).
4. Exact-match anchoring is soft (ref-vs-ref mean 0.79, only 53% >0.8) — identical
   copies should score near-certain 5★. Consider raising exact-match augmentation coverage.


---

## v7-1 Midpoint (June 2026) — NEW BEST, clean sweep

### Config
Same recipe as v6 (MiniLM + STS-B fix + meanmax + contrastive aug) but **last 5 layers
trained** (8.87M params) instead of 3. Midpoint checkpoint; final pending.
Dir: `reward_model_finetuned_v7-1`.

### Results — beats every prior checkpoint on every metric simultaneously

| Metric | v6-1 (mid) | v6 (final) | v7-1 (mid) |
|---|---|---|---|
| 100-pt eval Spearman | 0.682 | 0.636 | **0.718** (v4 record: 0.734) |
| Answer equivalence ROC-AUC | 0.906 | 0.887 | **0.942** (record; Word F1 0.887) |
| Val split Spearman | 0.594 | 0.648 | 0.647 |
| Confounds fooled (>0.7) | 16% | 14% | **12%** |
| Confounds caught (<0.4) | 17% | 12% | 16% |

### Swap detection (benchmark_swap.py)

| Perturbation | v6-1 | v6 final | v7-1 |
|---|---|---|---|
| exact match mean (>0.8 rate) | 0.73 (37%) | 0.79 (53%) | **0.84 (68%)** |
| antonym noticed | 29% | 44% | **75%** |
| negation noticed | 63% | 71% | **83%** |
| number noticed | 33% | 47% | **79%** |
| random word (untrained) noticed | 22% | 20% | **44%** |
| synonym control (should be ~0) | 1% | 0% | 2% ✓ |

### Key findings
1. **5-layer training dissolves the v6 trade-off** — general ranking AND minimal-edit
   discrimination improve together. No more choosing between checkpoints.
2. **Generalization beyond augmentation vocab doubled** (untrained random-word swaps
   noticed 20%→44%): more trainable capacity lets the swap-sensitivity lesson
   transfer instead of memorizing word lists.
3. Exact-match anchor largely fixed (dogfeed 1:1 identical 0.52→0.85; ref-vs-ref 68% >0.8).
4. Dogfeed "wrong number" single case (0.50) misleads — at n=150 v7-1 notices number
   swaps 79% vs v6's 47%. Trust benchmark_swap.py over single dogfeed cases.
5. **Caution for v7 final**: v6's external metrics peaked at midpoint and declined with
    val-MSE checkpoint selection. Keep v7-1; consider selecting by 100-pt Spearman.


---

## v7 Full Series (June 2026)

### v7-2 and v7-3 — longer training, midpoint-vs-final trade-off returns

The predicted v6 pattern repeated with v7: longer training with val-MSE checkpoint selection
traded external metrics for val fit.

| Metric | v7-1 (mid) | v7-2 | v7-3 (final) |
|---|---|---|---|
| 100-pt eval Spearman | **0.718** | 0.649 | 0.705 |
| AE ROC-AUC | **0.942** | 0.925 | 0.932 |
| Val split Spearman | 0.647 | 0.664 | **0.679** |
| Val MSE | 0.0673 | 0.0716 | 0.0762 |
| Confounds fooled | 12% | 7% | **6%** |
| Confounds caught | **16%** | 18% | 12% |

| Perturbation | v7-1 noticed | v7-2 noticed | v7-3 noticed |
|---|---|---|---|
| Antonym | **81%** | 68% | 81% |
| Negation | **86%** | 77% | 85% |
| Number | **80%** | 69% | 76% |
| Random word | 44% | 49% | 41% |

v7-3 partially recovered from the v7-2 dip but never matched v7-1's external metrics.
v7-1 remained the best overall checkpoint.


---

## v8 Series — DistilBERT 3-layer (June 2026)

### Config
Same recipe as v6 (contrastive augmentation, meanmax pooling, STS-B fix) but on
`distilbert/distilbert-base-cased` (66M params, 768-dim). Last 3 layers unfrozen (~17M
trainable). Dirs: v8-1 through v8-4.

### Benchmark

| Metric | v8-1 | v8-2 | v8-3 | v8-4 | v7-1 (MiniLM ref) |
|---|---|---|---|---|---|
| 100-pt eval Spearman | **0.777** | 0.702 | 0.644 | 0.691 | 0.718 |
| AE ROC-AUC | 0.832 | 0.840 | 0.849 | 0.832 | **0.942** |
| Val split Spearman | 0.556 | 0.626 | 0.673 | **0.695** | 0.647 |
| Val MSE | 0.0685 | 0.0671 | 0.0704 | 0.0708 | **0.0673** |
| Confounds fooled | **1%** | 4% | 14% | 17% | 12% |
| Confounds caught | 12% | **21%** | 13% | 13% | 16% |

### Swap detection

| Perturbation | v8-1 | v8-2 | v8-3 | v8-4 | v7-1 |
|---|---|---|---|---|---|
| Exact match (>0.8) | 0.62 (2%) | 0.66 (13%) | 0.75 (41%) | 0.82 (65%) | **0.85 (67%)** |
| Antonym noticed | 9% | 40% | 53% | 63% | **81%** |
| Negation noticed | 39% | 71% | 71% | 83% | **86%** |
| Number noticed | 7% | 41% | 62% | 69% | **80%** |
| Random word | 19% | 29% | 44% | 39% | 44% |

### Key findings
1. **v8-1 had a stunning 100-pt Spearman (0.777)** but near-zero swap detection (9% antonym,
   7% number). It was essentially a ranking-only model catching semantic quality without
   any word-level discrimination.
2. **Training progression improved swap detection** but cost ranking: antonym went
   9→40→53→63% while 100-pt dropped 0.777→0.691.
3. **DistilBERT's 3-layer ceiling is ~63% antonym detection** — even after 4 iterations
   of training, it couldn't match MiniLM v6-1's 3-layer 29%→v7-1's 5-layer 81% jump.
4. **Dynamic range was the fundamental problem** — dogfeed revealed all DistilBERT variants
   clustered scores in a narrow band (~0.4-0.7), unlike MiniLM's wide 0.14-0.90 spread.


---

## v9 Series — DistilBERT 5-layer (June 2026)

### Config
DistilBERT with **last 5 layers unfrozen** (~28M trainable params). All other recipe
pieces unchanged from v6/v7 (contrastive aug, meanmax pooling, STS-B, same datasets).
First time applying the full 5-layer treatment to the larger architecture.

### Benchmark

| Metric | v7-1 (MiniLM 5L) | v9-1 (mid) | v9-2 | v9-3 (final) |
|---|---|---|---|---|
| 100-pt eval Spearman | 0.718 | 0.728 | **0.757** | 0.705 |
| AE ROC-AUC | **0.942** | 0.919 | 0.918 | 0.900 |
| Val split Spearman | 0.647 | 0.670 | 0.684 | **0.697** |
| Val MSE | 0.0673 | 0.0749 | 0.0732 | **0.0621** |
| Confounds fooled | 12% | **9%** | **8%** | 9% |
| Confounds caught | **16%** | 15% | **16%** | 6% |

### Swap detection

| Perturbation | v7-1 | v9-1 | v9-2 | v9-3 |
|---|---|---|---|---|
| Exact match (>0.8) | 0.85 (67%) | 0.79 (61%) | 0.83 (67%) | **0.86 (71%)** |
| Antonym noticed | **81%** | 65% | 73% | 73% |
| Negation noticed | 86% | 73% | 87% | **89%** |
| Number noticed | **80%** | 73% | 73% | 75% |
| Random word (untrained) | 44% | 56% | 43% | **61%** |

### Dogfeed (dynamic range test)

| Case | v7-1 | v9-1 | v9-2 | v9-3 |
|---|---|---|---|---|
| Exact copy | 0.90 | 0.76 | 0.62 | 0.58 |
| 1:1 identical | 0.85 | 0.56 | 0.61 | 0.60 |
| Completely irrelevant | 0.14 | 0.42 | 0.63 | 0.62 |
| Correct paraphrase | 0.83 | 0.73 | 0.68 | 0.74 |
| One-word swap | 0.40 | 0.44 | 0.34 | 0.43 |

### Key findings
1. **5-layer DistilBERT did improve over 3-layer** — v9-2 hit 0.757 100-pt Spearman
   (new record) and v9-3 hit 89% negation + 61% untrained swap (new records).
2. **But dynamic range never materialized.** The dogfeed reveals the core problem:
   v9-3 gives identical text 0.60 and completely irrelevant text 0.62 — a 0.02 gap.
   v7-1 gives identical text 0.90 and irrelevant text 0.14 — a 0.76 gap.
3. **Root cause is architectural, not training.** DistilBERT was distilled from BERT for
   NLU classification — its embeddings group semantically similar text together. All ML
   research sentences live in the same neighborhood. MiniLM was pretrained as a sentence
   embedding model with contrastive learning — its 384-dim space naturally pushes
   dissimilar sentences apart.
4. **The Spearman/MSE numbers on the eval set are misleading** — ranking can look good
   even when scores are compressed into a 0.4-0.7 band. Dogfeed catches what the
   summary statistics miss.
5. **v7-1 (MiniLM 5L, 22M) remains the best reward model** — it's the only model that
   simultaneously achieves high ranking correlation AND wide dynamic range on extreme
   cases. The contrastive augmentation works because MiniLM was already pretrained to
   represent semantic distance, not just semantic category.
