"""Data augmentation for the reward model.

Two goals:
  1. Anchor exact-match / format signals (response == reference -> max score).
  2. Teach minimal-edit sensitivity: a one-word change that PRESERVES meaning
     must stay high, while a one-word change that FLIPS meaning must drop low.
     Without the meaning-preserving (synonym/filler) positives, the model just
     learns "any edit == bad", which is the wrong lesson and won't generalize.

Each transform takes (text, rng) and returns a new string, or None if it could
not be applied (e.g. no number / no swappable word present).
"""
import re

# ---- meaning-FLIPPING word pairs -> hard negatives ----
ANTONYM_PAIRS = [
    ("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"),
    ("higher", "lower"), ("more", "less"), ("larger", "smaller"), ("greater", "smaller"),
    ("faster", "slower"), ("better", "worse"), ("improves", "worsens"),
    ("positive", "negative"), ("above", "below"), ("before", "after"),
    ("maximum", "minimum"), ("always", "never"), ("all", "none"),
    ("true", "false"), ("present", "absent"), ("enabled", "disabled"),
    ("significant", "insignificant"), ("same", "different"), ("similar", "different"),
    ("accept", "reject"), ("supports", "contradicts"), ("with", "without"),
]
# ---- meaning-PRESERVING word pairs -> positives ----
SYNONYM_PAIRS = [
    ("big", "large"), ("small", "tiny"), ("fast", "quick"), ("method", "approach"),
    ("shows", "demonstrates"), ("uses", "utilizes"), ("helps", "aids"),
    ("important", "crucial"), ("result", "outcome"), ("however", "but"),
    ("therefore", "thus"), ("approximately", "about"), ("because", "since"),
    ("obtain", "get"), ("multiple", "several"), ("rise", "increase"),
    ("accurate", "precise"), ("aim", "goal"), ("begin", "start"), ("end", "finish"),
]
FILLERS = ["In summary, ", "Notably, ", "As shown, ", "Overall, ",
           "In other words, ", "Essentially, ", "To clarify, "]
AUX_VERBS = ["is", "are", "was", "were", "can", "will", "does", "do",
             "has", "have", "should", "could", "would"]


def _build_map(pairs):
    m = {}
    for a, b in pairs:
        m.setdefault(a, b)
        m.setdefault(b, a)
    return m


ANTONYM_MAP = _build_map(ANTONYM_PAIRS)
SYNONYM_MAP = _build_map(SYNONYM_PAIRS)


def _swap_one(text, mapping, rng):
    """Replace the first occurrence of a random swappable word, preserving case."""
    words = re.findall(r"[A-Za-z]+", text)
    cands = [w for w in words if w.lower() in mapping]
    if not cands:
        return None
    w = rng.choice(cands)
    repl = mapping[w.lower()]
    if w[0].isupper():
        repl = repl[0].upper() + repl[1:]
    return re.sub(r"\b" + re.escape(w) + r"\b", repl, text, count=1)


# ---- positives (meaning preserved) ----
def synonym_swap(text, rng):
    return _swap_one(text, SYNONYM_MAP, rng)


def filler_insert(text, rng):
    return rng.choice(FILLERS) + text


# ---- hard negatives (meaning flipped) ----
def antonym_swap(text, rng):
    return _swap_one(text, ANTONYM_MAP, rng)


def negation_flip(text, rng):
    """Remove an existing 'not' (de-negate) or insert one after an auxiliary verb."""
    if re.search(r"\bnot\b", text):
        return re.sub(r"\s*\bnot\b", "", text, count=1)
    for aux in AUX_VERBS:
        m = re.search(r"\b" + aux + r"\b", text)
        if m:
            return text[: m.end()] + " not" + text[m.end():]
    return None


def number_swap(text, rng):
    nums = re.findall(r"\b\d+[\d,.]*%?\b", text)
    if not nums:
        return None
    num = rng.choice(nums)
    raw = num.replace(",", "").replace("%", "").rstrip(".")
    try:
        val = float(raw)
    except ValueError:
        return None
    new_val = val * rng.choice([0.5, 0.6, 0.7, 1.3, 1.5, 1.7, 2.0])
    new_str = str(int(new_val)) if new_val == int(new_val) else f"{new_val:.1f}"
    if num.endswith("%"):
        new_str += "%"
    return re.sub(r"\b" + re.escape(num) + r"\b", new_str, text, count=1)


# ---- coarse negatives (length / coherence) ----
def truncate_half(text, rng):
    words = text.split()
    if len(words) < 6:
        return None
    return " ".join(words[: len(words) // 2])


def duplicate(text, rng):
    return f"{text} {text}"


def word_shuffle(text, rng):
    words = text.split()
    if len(words) < 6:
        return None
    shuffled = words[:]
    rng.shuffle(shuffled)
    if shuffled == words:
        return None
    return " ".join(shuffled)


# Registry: (transform, target_score on the 1-5 scale).
# Positives stay high; semantic confounds sit at 3.0 (a fluent but wrong answer)
# to match the rest of the dataset's scale.
POSITIVE_TRANSFORMS = [
    (synonym_swap, 4.5),
    (filler_insert, 4.5),
]
NEGATIVE_TRANSFORMS = [
    (antonym_swap, 3.0),
    (negation_flip, 3.0),
    (number_swap, 3.0),
    (truncate_half, 2.0),
    (duplicate, 2.0),
    (word_shuffle, 1.5),
]
ALL_TRANSFORMS = POSITIVE_TRANSFORMS + NEGATIVE_TRANSFORMS


def build_augmentations(ref_pool, rng, cap_per_type=1500, min_words=4):
    """Build appended augmentation records from a pool of gold reference strings.

    For each reference we also emit an exact-match (5.0) anchor. Returns a list of
    {orig_reference_answer, orig_response, orig_score} dicts.
    """
    records = []
    counts = {fn.__name__: 0 for fn, _ in ALL_TRANSFORMS}
    counts["exact_match"] = 0

    for ref in ref_pool:
        if counts["exact_match"] < cap_per_type:
            records.append({"orig_reference_answer": ref, "orig_response": ref, "orig_score": 5.0})
            counts["exact_match"] += 1
        for fn, score in ALL_TRANSFORMS:
            if counts[fn.__name__] >= cap_per_type:
                continue
            out = fn(ref, rng)
            if out and out != ref and len(out.split()) >= min_words:
                records.append({"orig_reference_answer": ref, "orig_response": out, "orig_score": score})
                counts[fn.__name__] += 1
        if all(c >= cap_per_type for c in counts.values()):
            break

    return records, counts
