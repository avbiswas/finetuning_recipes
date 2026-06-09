"""Quick experiment: batch-level exact-match augmentation."""
import sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"
BATCH, LR = 32, 1e-4
EPOCHS, MAX_LEN = 3, 256

class MiniDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        return r["text"], (r["score"] - 1.0) / 4.0

def test_exact(model, tokenizer, refs, name):
    scores = []
    for i in range(0, len(refs), 64):
        batch = refs[i:i+64]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            preds = model(enc["input_ids"], enc["attention_mask"])
            scores.extend(preds.tolist())
    s = np.clip(np.array(scores), 0, 1)
    print(f"  {name}: mean={s.mean():.4f} median={np.median(s):.4f} >0.8={(s>0.8).sum()}/{len(s)}")

def train(with_aug=False):
    """Train and return model. with_aug = apply batch-level exact-match augmentation."""
    ds = load_dataset("paperbd/paper_answers_reward", split="train")
    records = [dict(r) for r in ds][:2000]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    for r in records:
        r["text"] = f"{r['orig_reference_answer']} [SEP] {r['orig_response']}"
        r["score"] = r["orig_score"]
    
    loader = DataLoader(MiniDataset(records), batch_size=BATCH, shuffle=True)
    
    model = nn.Sequential(
        AutoModel.from_pretrained(MODEL_ID),
    ).to(DEVICE)
    # Simple: just finetune last layer embeddings via a head (quick test)
    class SimpleHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(MODEL_ID)
            for p in self.encoder.parameters(): p.requires_grad = False
            self.head = nn.Linear(self.encoder.config.hidden_size, 1)
        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return self.head(pooled).squeeze(-1)
    
    model = SimpleHead().to(DEVICE)
    opt = torch.optim.AdamW(model.head.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        for texts, labels in loader:
            texts = list(texts)
            if with_aug:
                # Batch-level: 20% of batch gets response = reference, label = 5.0
                for i in range(len(texts)):
                    if random.random() < 0.2:
                        ref = texts[i].split(" [SEP] ")[0]
                        texts[i] = f"{ref} [SEP] {ref}"
                        labels[i] = 1.0
            
            enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            preds = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
            loss = F.mse_loss(preds, labels.float().to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    
    return model, tokenizer

# Test refs: 50 random references from test set
test_ds = load_dataset("paperbd/paper_answers_reward", split="test")
import random; rng = random.Random(42)
test_refs = rng.sample(list(set(r["orig_reference_answer"] for r in test_ds if isinstance(r["orig_reference_answer"], str))), 50)

print("Training without augmentation...")
m1, tok = train(with_aug=False)
test_exact(m1, tok, test_refs, "No aug")

print("\nTraining WITH batch-level exact-match augmentation...")
m2, tok = train(with_aug=True)
test_exact(m2, tok, test_refs, "With aug")
