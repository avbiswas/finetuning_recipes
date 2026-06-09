"""
Procedural confounds: keyword swaps, number swaps, verb flips.
"""
import argparse, json, random, re, os
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

VERB_PAIRS = [
    ("increases", "decreases"), ("decreases", "increases"),
    ("improves", "worsens"), ("outperforms", "underperforms"),
    ("proposed", "rejected"), ("achieves", "fails to achieve"),
    ("enables", "prevents"), ("reduces", "increases"),
    ("higher", "lower"), ("better", "worse"),
    ("more", "fewer"), ("larger", "smaller"),
    ("faster", "slower"), ("novel", "standard"),
]


def number_swap(text, rng):
    nums = re.findall(r'\b\d+[\d,.]*%?\b', text)
    if not nums:
        return None
    num = rng.choice(nums)
    raw = num.replace(",", "").replace("%", "").rstrip(".")
    try:
        val = float(raw)
    except ValueError:
        return None
    factor = rng.choice([0.5, 0.6, 0.7, 1.3, 1.5, 1.7, 2.0])
    new_val = val * factor
    new_str = str(int(new_val)) if new_val == int(new_val) else f"{new_val:.1f}"
    if num.endswith("%"):
        new_str += "%"
    return re.sub(r'\b' + re.escape(num) + r'\b', new_str, text, count=1)


def verb_swap(text, rng):
    verbs = rng.sample(VERB_PAIRS, len(VERB_PAIRS))
    text_lower = text.lower()
    for a, b in verbs:
        if a in text_lower:
            idx = text_lower.index(a)
            return text[:idx] + b + text[idx + len(a):]
    return None


def keyword_swap(text, vectorizer, tfidf, names, rng):
    """Swap a top-TFIDF entity keyword with random corpus entity."""
    words = re.findall(r'\b\w+\b', text)
    entity_words = [w for w in words if w[0].isupper() and len(w) > 1 and not w.isdigit()]
    if not entity_words:
        return None
    # Pick one at random from the entity words
    original = rng.choice(entity_words)
    # Pick a replacement from corpus entities
    candidates = [e for e in entity_set if e.lower() != original.lower() and e.isalpha()]
    if not candidates:
        return None
    replacement = rng.choice(candidates)
    return re.sub(r'\b' + re.escape(original) + r'\b', replacement, text, count=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="data/rewards_dataset")
    args = parser.parse_args()

    rng = random.Random(42)

    ds = load_dataset("paperbd/paper_answers_reward", split="train")
    refs = [r["orig_reference_answer"] if isinstance(r["orig_reference_answer"], str) else json.dumps(r["orig_reference_answer"]) for r in ds]
    fivestar = [(i, r) for i, r in enumerate(ds) if r["orig_score"] == 5.0 and len(str(r["orig_response"]).split()) > 8]
    rng.shuffle(fivestar)
    print(f"5★ responses available: {len(fivestar)}")

    # Build TF-IDF keywords and entity set for keyword swaps
    print("Building TF-IDF...")
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=3, max_features=10000, lowercase=True)
    tfidf = vectorizer.fit_transform(refs)

    # Corpus entities (for replacements)
    entity_set = set()
    for ref in refs:
        for w in re.findall(r'\b[A-Z][a-z]+(?:-\d+)?\b|\b[A-Z]{2,}\b', ref):
            entity_set.add(w)

    methods = {
        "keyword_swap": 400,
        "number_swap": 400,
        "verb_swap": 400,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for method, target in methods.items():
        confounds = []
        attempts = 0
        max_attempts = len(fivestar) * 3

        for idx, r in fivestar:
            if len(confounds) >= target:
                break
            ref = refs[idx]
            resp = str(r["orig_response"])
            attempts += 1

            if method == "keyword_swap":
                swapped = keyword_swap(resp, vectorizer, tfidf, None, rng)
            elif method == "number_swap":
                swapped = number_swap(resp, rng)
            elif method == "verb_swap":
                swapped = verb_swap(resp, rng)
            else:
                swapped = None

            if swapped and swapped != resp and len(swapped.split()) > 3:
                confounds.append({
                    "orig_reference_answer": ref,
                    "orig_response": swapped,
                    "orig_score": 3.0,
                })

        outpath = f"{args.output_dir}/confounds_{method}.jsonl"
        with open(outpath, "w") as f:
            for r in confounds:
                f.write(json.dumps(r) + "\n")
        print(f"  {method}: {len(confounds)}/{target} ({attempts} attempts) → {outpath}")

    # Split into train/test (75/25)
    all_confounds = []
    for method in methods:
        path = f"{args.output_dir}/confounds_{method}.jsonl"
        with open(path) as f:
            all_confounds.extend(json.loads(l) for l in f)
    rng.shuffle(all_confounds)

    split = int(len(all_confounds) * 0.75)
    train_c = all_confounds[:split]
    test_c = all_confounds[split:]

    for name, data in [("train", train_c), ("test", test_c)]:
        outpath = f"{args.output_dir}/procedural_confounds_{name}.jsonl"
        with open(outpath, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        print(f"  {name}: {len(data)} records → {outpath}")
