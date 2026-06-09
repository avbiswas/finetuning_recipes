import json
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment import build_augmentations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default=None, help="Output model name suffix")
args = parser.parse_args()

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL_ID = "distilbert/distilbert-base-cased"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
LR_HEAD = 5e-4
LR_ENCODER = 5e-5
DROPOUT = 0.2
EPOCHS = 50
MAX_LEN = 512
PATIENCE = 5
UNFREEZE_LAYERS = 3  # last N transformer layers to unfreeze. 0 = head only, 6 = all
POOLING = "meanmax"  # "mean" | "max" | "meanmax" — meanmax keeps mean (what MiniLM
#                       was trained for) + max (preserves single changed-token signal)
OUTPUT_DIR = f"models/reward_model_finetuned_{args.output}" if args.output else "models/reward_model_finetuned"


def get_encoder_layers(encoder):
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
        return encoder.transformer.layer
    raise AttributeError(f"Unsupported encoder layout for {encoder.__class__.__name__}")


class RewardDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return r["text"], (r["score"] - 1.0) / 4.0


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_ID)
        for p in self.encoder.parameters():
            p.requires_grad = False
        if UNFREEZE_LAYERS > 0:
            for layer in get_encoder_layers(self.encoder)[-UNFREEZE_LAYERS:]:
                for p in layer.parameters():
                    p.requires_grad = True
        pool_dim = self.encoder.config.hidden_size * (2 if POOLING == "meanmax" else 1)
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(pool_dim, 1),
        )

    def pool(self, hidden, mask):
        mask_f = mask.unsqueeze(-1).float()
        mean = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        if POOLING == "mean":
            return mean
        # mask padded positions to -inf so they never win the max
        mx = hidden.masked_fill(mask_f == 0, float("-inf")).max(1).values
        if POOLING == "max":
            return mx
        return torch.cat([mean, mx], dim=-1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(out.last_hidden_state, attention_mask)
        return self.head(pooled).squeeze(-1)

    def score_batch(self, references, responses):
        """Batch reward scoring. references and responses are lists of strings."""
        texts = [f"{r} [SEP] {c}" for r, c in zip(references, responses)]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.eval()
        with torch.no_grad():
            scores = self(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        return scores.cpu().tolist()


# ---- Load data ----
print("Loading data...")
import random
rng = random.Random(42)

# Primary: our uploaded science dataset
science = load_dataset("paperbd/paper_answers_reward")
train_records = list(science["train"])
val_records = list(science["test"])

# Mix in Feedback-Collection (10K)
fc = load_dataset("prometheus-eval/Feedback-Collection", split="train")
fc_idx = rng.sample(range(len(fc)), 10000)
for i in fc_idx:
    r = fc[int(i)]
    train_records.append({
        "orig_reference_answer": r["orig_reference_answer"],
        "orig_response": r["orig_response"],
        "orig_score": float(r["orig_score"]),
    })

# Mix in answer equivalence (9K)
ae = load_dataset("kortukov/answer-equivalence-dataset", split="train")
for r in ae:
    train_records.append({
        "orig_reference_answer": r["reference"],
        "orig_response": r["candidate"],
        "orig_score": 4.0 if r["label"] == "equivalent" else 1.5,
    }    )

# Mix in STS-B (sentence similarity, scored 0-5)
stsb = load_dataset("sentence-transformers/stsb", split="train")
for r in stsb:
    train_records.append({
        "orig_reference_answer": r["sentence1"],
        "orig_response": r["sentence2"],
        "orig_score": float(r["score"]) * 4.0 + 1.0,  # stsb score is 0-1, map to 1-5
    })

rng.shuffle(train_records)

# ---- Synthetic data (appended) ----
# Minimal-edit contrastive pairs built on the GOLD reference text: a one-word
# change that preserves meaning (synonym/filler) stays high (4.5), while a
# one-word change that flips meaning (antonym/negation/number) drops to 1.5.
# This is what teaches the model to attend to WHICH word changed instead of
# just edit distance. Also includes exact-match (5.0) and coarse length/format
# negatives. See augment.py for the transforms.
ref_pool = list({
    r["orig_reference_answer"]
    for r in train_records
    if isinstance(r.get("orig_reference_answer"), str) and len(r["orig_reference_answer"].split()) >= 8
})
rng.shuffle(ref_pool)

aug_records, aug_counts = build_augmentations(ref_pool, rng, cap_per_type=1500)
train_records.extend(aug_records)
print(f"Appended {len(aug_records)} augmented records: {aug_counts}")

rng.shuffle(train_records)
print(f"Train: {len(train_records)}, Val: {len(val_records)}")

# Tokenize
print("Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
for rec in train_records + val_records:
    rec["text"] = f"{rec['orig_reference_answer']} [SEP] {rec['orig_response']}"
    rec["score"] = rec["orig_score"]

train_ds = RewardDataset(train_records)
val_ds = RewardDataset(val_records)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ---- Train ----
print("Training...")
model = RewardModel().to(DEVICE)

encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
head_params = list(model.head.parameters())
trainable = sum(p.numel() for p in encoder_params) + sum(p.numel() for p in head_params)
print(f"Trainable params: {trainable:,} (last 3 layers + head)")
print(f"LR: head={LR_HEAD}, encoder={LR_ENCODER}, dropout={DROPOUT}")

optimizer = torch.optim.AdamW(
    [
        {"params": head_params, "lr": LR_HEAD},
        {"params": encoder_params, "lr": LR_ENCODER},
    ]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

best_val_mse = float("inf")
best_state = None
patience_counter = 0
VAL_EVERY = 250
global_step = 0

pbar = tqdm(total=EPOCHS * len(train_loader), desc="Training")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_idx, (texts, labels) in enumerate(train_loader):
        texts = list(texts)
        # Augmentation (exact-match + minimal-edit contrastive pairs) is now
        # appended to the dataset offline; see build_augmentations above.
        enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)
        labels_batch = labels.float().to(DEVICE)

        optimizer.zero_grad()
        preds = model(input_ids, mask)
        loss = F.mse_loss(preds, labels_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        global_step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "best_mse": f"{best_val_mse:.4f}"})

        # Validate every VAL_EVERY steps
        if global_step % VAL_EVERY == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for v_texts, v_labels in val_loader:
                    enc_v = tokenizer(v_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                    v_preds = model(enc_v["input_ids"].to(DEVICE), enc_v["attention_mask"].to(DEVICE))
                    v_loss = F.mse_loss(v_preds, v_labels.float().to(DEVICE))
                    val_loss += v_loss.item()

            val_mse = val_loss / len(val_loader)

            improved = val_mse < best_val_mse
            if improved:
                best_val_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                model.encoder.save_pretrained(OUTPUT_DIR)
                tokenizer.save_pretrained(OUTPUT_DIR)
                head_state = {k.replace("head.", ""): v for k, v in best_state.items() if k.startswith("head.")}
                torch.save(head_state, f"{OUTPUT_DIR}/head_weights.pt")
                patience_counter = 0
                tqdm.write(f"Step {global_step}: val_mse={val_mse:.4f} ✓ saved")
            else:
                patience_counter += 1
                tqdm.write(f"Step {global_step}: val_mse={val_mse:.4f} (no improvement {patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                tqdm.write(f"Early stopping at step {global_step}")
                break
            model.train()

    if patience_counter >= PATIENCE:
        break

pbar.close()

# ---- Restore best weights ----
print(f"Restoring best model (val_mse={best_val_mse:.4f})...")
model.load_state_dict(best_state)

# ---- Quick test ----
print("\nTesting batch scoring...")
refs = ["The capital of France is Paris.", "2 + 2 = 4"]
resps = ["Paris is the capital of France.", "2 + 2 equals 4"]
scores = model.score_batch(refs, resps)
for r, c, s in zip(refs, resps, scores):
    print(f"  Ref: {r}")
    print(f"  Hyp: {c}")
    print(f"  Score: {s:.4f}")
    print()
