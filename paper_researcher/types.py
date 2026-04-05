from dataclasses import dataclass


@dataclass
class QAPair:
    question: str
    answer: str

    def __repr__(self):
        return f"QAPair(question={self.question!r}, answer={self.answer!r})"


@dataclass
class Triplet:
    subject: str
    relation: str
    object: str

    def __repr__(self):
        return f"({self.subject}, {self.relation}, {self.object})"


@dataclass
class RetrievalResult:
    index: int | None   # 0-based index into passages list; None if no passage answers
    reasoning: str
    raw: str            # full model output

    def __repr__(self):
        return f"RetrievalResult(index={self.index}, reasoning={self.reasoning!r})"
