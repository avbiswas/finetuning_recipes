import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# BASE_MODEL = "distilbert/distilbert-base-cased"

EMBED_DIM = None  # auto-detected from config if None


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def load_reward_model(path):
    """
    Load a trained MiniLM reward model.

    Usage:
        model, tokenizer = load_reward_model("paperbd/neuraltxt-reward-22M")
        score = model.score("reference text", "candidate text")
    """
    path = Path(path).resolve()

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Encoder
    if (path / "model.safetensors").exists():
        encoder = AutoModel.from_pretrained(str(path))
    else:
        encoder = AutoModel.from_pretrained(BASE_MODEL)

    # Head (~3KB)
    # Auto-detect embedding dim
    dim = EMBED_DIM or encoder.config.hidden_size
    head = nn.Sequential(nn.Dropout(0.1), nn.Linear(dim, 1))
    for fname in ["head_weights.pt", "head_weights.bin"]:
        hp = path / fname
        if hp.exists():
            head.load_state_dict(torch.load(str(hp), weights_only=True, map_location="cpu"))
            break
    head.eval()

    class RewardScorer:
        def __init__(self):
            self.encoder = encoder
            self.head = head

        def score(self, reference, response):
            text = f"{reference} [SEP] {response}"
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                pooled = mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                return self.head(pooled).item()

        def score_batch(self, references, responses):
            texts = [f"{r} [SEP] {c}" for r, c in zip(references, responses)]
            enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                pooled = mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                return self.head(pooled).squeeze(-1).tolist()

    return RewardScorer(), tokenizer
