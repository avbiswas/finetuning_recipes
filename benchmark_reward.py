"""Benchmark a reward model across 3 datasets."""
import sys, json, torch, torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "reasoning_train")
from reward_model import load_reward_model

MODEL_DIR = "models/reward_model_final/reward_model_finetuned_v5"
MAX_LEN = 512


def load_our_model(path):
    return load_reward_model(path)[0]


def word_f1(ref, hyp):
    rt = set(ref.lower().split())
    ht = set(hyp.lower().split())
    if not ht:
        return 0.0
    p = len(rt & ht) / len(ht)
    rv = len(rt & ht) / len(rt)
    return 2 * p * rv / (p + rv) if (p + rv) > 0 else 0.0


# ===== 1. 100-point eval =====
print("=" * 60)
print("1. 100-POINT EVAL (sft_v2 responses)")
print("-" * 60)

records = []
with open("instruction_tuning/evals/smollm_135M_neuraltxt_sft_v2_mlx_results.jsonl") as f:
    for line in f:
        records.append(json.loads(line))
refs = [r["ground_truth"] for r in records]
resps = [r["response"] for r in records]

judge_records = []
with open("instruction_tuning/evals/smollm_135M_neuraltxt_sft_v2_mlx_results_judged.jsonl") as f:
    for line in f:
        judge_records.append(json.loads(line))
gt = np.array([sum(r["scores"].values()) / 20.0 for r in judge_records])

our = load_our_model(MODEL_DIR)
our_scores = our.score_batch(refs, resps)

from qa_metrics.RewardBert import RewardBert
rb = RewardBert(device="cpu")
rb_scores = [rb.compute_score(refs[i][:500], resps[i][:500])[0] for i in range(len(refs))]
f1_scores = [word_f1(refs[i], resps[i]) for i in range(len(refs))]

for name, scores in [("Ours", our_scores), ("RewardBert", rb_scores), ("Word F1", f1_scores)]:
    sr, _ = spearmanr(gt, scores)
    print(f"  {name:<12} Spearman r={sr:.4f}  Mean={np.mean(scores):.4f}")

# ===== 2. Validation confounds =====
print(f"\n{'='*60}")
print("2. VALIDATION CONFOUNDS")
print("-" * 60)

val = load_dataset("paperbd/paper_answers_reward", split="test")
confounds = [r for r in val if r["orig_score"] == 3.0]
refs_c = [r["orig_reference_answer"] for r in confounds]
bads_c = [r["orig_response"] for r in confounds]

our_c = our.score_batch(refs_c, bads_c)
rb_c = [rb.compute_score(str(refs_c[i])[:500], str(bads_c[i])[:500])[0] for i in range(len(refs_c))]
f1_c = [word_f1(str(refs_c[i]), str(bads_c[i])) for i in range(len(refs_c))]

for name, scores in [("Ours", our_c), ("RewardBert", rb_c), ("Word F1", f1_c)]:
    s = np.array(scores)
    fooled = (s > 0.7).sum()
    caught = (s < 0.4).sum()
    print(f"  {name:<12} Mean={s.mean():.4f}  Fooled(>0.7)={fooled}/{len(s)} ({100*fooled/len(s):.0f}%)  Caught(<0.4)={caught}/{len(s)} ({100*caught/len(s):.0f}%)")

# ===== 3. Answer equivalence =====
print(f"\n{'='*60}")
print("3. ANSWER EQUIVALENCE")
print("-" * 60)

ae = load_dataset("kortukov/answer-equivalence-dataset", split="dev")
ae_refs = ae["reference"]
ae_cands = ae["candidate"]
ae_labels = [1 if l == "equivalent" else 0 for l in ae["label"]]

our_ae = our.score_batch(ae_refs, ae_cands)
rb_ae = [rb.compute_score(ae_refs[i][:500], ae_cands[i][:500])[0] for i in range(len(ae_refs))]
f1_ae = [word_f1(ae_refs[i], ae_cands[i]) for i in range(len(ae_refs))]

for name, scores in [("Ours", our_ae), ("RewardBert", rb_ae), ("Word F1", f1_ae)]:
    auc = roc_auc_score(ae_labels, scores)
    s = np.array(scores)
    eq = np.array(ae_labels) == 1
    print(f"  {name:<12} ROC-AUC={auc:.4f}  Equiv mean={s[eq].mean():.4f}  Not mean={s[~eq].mean():.4f}")

# ===== 4. Val set from our dataset =====
print(f"\n{'='*60}")
print("4. VALIDATION (paperbd/papers_answer_equivalance test split)")
print("-" * 60)

vrefs = val["orig_reference_answer"]
vresps = val["orig_response"]
vlabels = [s / 5.0 for s in val["orig_score"]]

our_val = our.score_batch(vrefs, vresps)
sr_v, _ = spearmanr(vlabels, our_val)
print(f"  Spearman r={sr_v:.4f}  Mean pred={np.mean(our_val):.4f}  Mean label={np.mean(vlabels):.4f}")
