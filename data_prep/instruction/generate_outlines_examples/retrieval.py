import json
import random

from pydantic import BaseModel, Field

from model import generate_structured_output
from utils import AlpacaDataset


class UniqueQuestions(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=4)


class RetrievalReason(BaseModel):
    why: str = Field(..., max_length=300)


class NoAnswerReason(BaseModel):
    why: str = Field(..., max_length=300)


QUESTION_EXTRACTION_PROMPT = """
You will be given one passage.

Extract a small set of unique questions that can be answered directly from the passage.
Keep the questions short and specific.
Avoid vague questions such as "What is described?", "What does this talk about?", or similar generic wording.
Do not ask duplicate questions.
Do not ask questions that require outside knowledge.
Only return questions whose answers are clearly present in the passage.
"""


RETRIEVAL_REASON_PROMPT = """
You will be given one passage and one question.

Write a short reason explaining why the passage answers the question.
Keep the reason grounded in the passage.
Do not mention any passage numbers.
Do not use bullet points or JSON.
Be concise and specific.
"""


NO_ANSWER_REASON_PROMPT = """
You will be given several passages and one question.

Write a short reason explaining why none of the passages answer the question.
Keep the reason grounded in the passages.
Do not mention passage numbers.
Do not use bullet points or JSON.
Be concise and specific.
"""


RETRIEVAL_INSTRUCTION = (
    "Read the passages and identify which passage answers the question. "
    "Return the passage number and a short justification in markdown."
)


RETRIEVAL_JSON_INSTRUCTION = (
    "Read the passages and identify which passage answers the question. "
    "Return a JSON object with keys 'passage' and 'why'. "
    "Use a passage number when an answer is present, or null when none of the passages answer the question."
)


def extract_unique_questions(passage: str) -> UniqueQuestions:
    messages = [
        {"role": "system", "content": QUESTION_EXTRACTION_PROMPT},
        {"role": "user", "content": passage},
    ]
    return generate_structured_output(messages, UniqueQuestions, temp=0.2)


def generate_retrieval_reason(passage: str, question: str) -> RetrievalReason:
    messages = [
        {"role": "system", "content": RETRIEVAL_REASON_PROMPT},
        {
            "role": "user",
            "content": f"Passage:\n{passage}\n\nQuestion: {question}",
        },
    ]
    return generate_structured_output(messages, RetrievalReason, temp=0.2)


def generate_no_answer_reason(passages: list[str], question: str) -> NoAnswerReason:
    messages = [
        {"role": "system", "content": NO_ANSWER_REASON_PROMPT},
        {
            "role": "user",
            "content": f"{format_passages(passages)}\n\nQuestion: {question}",
        },
    ]
    return generate_structured_output(messages, NoAnswerReason, temp=0.2)


def format_passages(passages: list[str]) -> str:
    return "\n\n".join(
        f"Passage {idx}:\n{passage}"
        for idx, passage in enumerate(passages, start=1)
    )


def build_retrieval_input(passages: list[str], question: str) -> str:
    return f"{format_passages(passages)}\n\nQuestion: {question}"


def build_retrieval_output(passage_index: int, reason: str) -> str:
    return f"**Passage:** {passage_index + 1}\n\n**Why:** {reason}"


def build_no_answer_output(reason: str) -> str:
    return f"**Passage:** None\n\n**Why:** {reason}"


def build_retrieval_json_output(passage_index: int, reason: str) -> str:
    return json.dumps(
        {"passage": passage_index + 1, "why": reason},
        ensure_ascii=False,
    )


def build_no_answer_json_output(reason: str) -> str:
    return json.dumps(
        {"passage": None, "why": reason},
        ensure_ascii=False,
    )


def convert_to_dataset(
    passages: list[str],
    extracted_questions: list[UniqueQuestions],
) -> list[AlpacaDataset]:
    dataset = []

    for correct_idx, questions_for_passage in enumerate(extracted_questions):
        for question in questions_for_passage.questions:
            positive_reason = generate_retrieval_reason(passages[correct_idx], question)

            full_input = build_retrieval_input(passages, question)
            dataset.append(
                AlpacaDataset(
                    instruction=RETRIEVAL_INSTRUCTION,
                    input=full_input,
                    output=build_retrieval_output(correct_idx, positive_reason.why),
                )
            )
            dataset.append(
                AlpacaDataset(
                    instruction=RETRIEVAL_JSON_INSTRUCTION,
                    input=full_input,
                    output=build_retrieval_json_output(correct_idx, positive_reason.why),
                )
            )

            negative_passages = [
                passage
                for idx, passage in enumerate(passages)
                if idx != correct_idx
            ]
            if not negative_passages:
                continue

            negative_reason = generate_no_answer_reason(negative_passages, question)
            negative_input = build_retrieval_input(negative_passages, question)
            dataset.append(
                AlpacaDataset(
                    instruction=RETRIEVAL_INSTRUCTION,
                    input=negative_input,
                    output=build_no_answer_output(negative_reason.why),
                )
            )
            dataset.append(
                AlpacaDataset(
                    instruction=RETRIEVAL_JSON_INSTRUCTION,
                    input=negative_input,
                    output=build_no_answer_json_output(negative_reason.why),
                )
            )

    return dataset


def main(passages: list[str]) -> list[AlpacaDataset]:
    cleaned_passages = [passage.strip() for passage in passages if passage.strip()]
    if len(cleaned_passages) < 2:
        return []

    shuffled_passages = cleaned_passages[:]
    random.shuffle(shuffled_passages)

    extracted_questions = [
        extract_unique_questions(passage)
        for passage in shuffled_passages
    ]

    return convert_to_dataset(shuffled_passages, extracted_questions)


if __name__ == "__main__":
    dataset = main(
        [
            """
The Transformer replaces recurrence with attention mechanisms and achieves strong translation results while improving parallelization.
            """,
            """
Convolutional networks apply learned filters over local neighborhoods and are widely used in computer vision tasks.
            """,
            """
Retrieval-augmented generation combines a retriever with a generator so the model can ground its response in external documents.
            """,
        ]
    )

    print(len(dataset))
