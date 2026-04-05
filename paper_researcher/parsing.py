"""
Best-effort parsers for structured model outputs.
All parsers fall back gracefully — never raise.
"""
import re
from .types import QAPair, Triplet, RetrievalResult


def parse_bullets(text: str) -> list[str]:
    lines = text.strip().splitlines()
    bullets = []
    for line in lines:
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            bullets.append(line[2:].strip())
    return bullets if bullets else [text.strip()]


def parse_qa_pairs(text: str) -> list[QAPair]:
    """Parse markdown Q&A blocks: **Question:** ... **Answer:** ..."""
    pairs = []
    # Match ### Q1 ... blocks
    blocks = re.split(r"###\s*Q\d+", text)
    for block in blocks:
        q_match = re.search(r"\*\*Question:\*\*\s*(.+?)(?=\*\*Answer:|$)", block, re.DOTALL)
        a_match = re.search(r"\*\*Answer:\*\*\s*(.+?)(?=###|$)", block, re.DOTALL)
        if q_match and a_match:
            pairs.append(QAPair(
                question=q_match.group(1).strip(),
                answer=a_match.group(1).strip(),
            ))

    # Fallback: plain Q: / A: pattern
    if not pairs:
        q_matches = re.findall(r"(?:Q:|Question:)\s*(.+?)(?=A:|Answer:|$)", text, re.DOTALL | re.IGNORECASE)
        a_matches = re.findall(r"(?:A:|Answer:)\s*(.+?)(?=Q:|Question:|$)", text, re.DOTALL | re.IGNORECASE)
        for q, a in zip(q_matches, a_matches):
            pairs.append(QAPair(question=q.strip(), answer=a.strip()))

    return pairs if pairs else [QAPair(question="", answer=text.strip())]


def parse_triplets(text: str) -> list[Triplet]:
    """Parse (subject, relation, object) from markdown bullets."""
    triplets = []
    for match in re.finditer(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", text):
        triplets.append(Triplet(
            subject=match.group(1).strip(),
            relation=match.group(2).strip(),
            object=match.group(3).strip(),
        ))
    return triplets if triplets else []


def parse_retrieval(text: str, num_passages: int) -> RetrievalResult:
    """Parse **Passage:** N and **Why:** ... from model output."""
    passage_match = re.search(r"\*\*Passage:\*\*\s*(\w+)", text)
    why_match = re.search(r"\*\*Why:\*\*\s*(.+?)(?=$)", text, re.DOTALL)

    reasoning = why_match.group(1).strip() if why_match else text.strip()

    if passage_match:
        val = passage_match.group(1).strip()
        if val.lower() == "none":
            return RetrievalResult(index=None, reasoning=reasoning, raw=text)
        try:
            idx = int(val) - 1  # convert to 0-based
            idx = idx if 0 <= idx < num_passages else None
            return RetrievalResult(index=idx, reasoning=reasoning, raw=text)
        except ValueError:
            pass

    return RetrievalResult(index=None, reasoning=reasoning, raw=text)
