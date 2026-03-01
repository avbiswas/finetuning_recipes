from pydantic import BaseModel, Field

from model import augment_data, generate_structured_output
from utils import AlpacaDataset


class BulletList(BaseModel):
    bullets: list[str] = Field(..., min_length=1, max_length=6)


SYSTEM_PROMPT = """
Given this passage, extract a short list of important bullet points.
Keep each point concise and grounded in the passage.
Do not introduce information that is not present in the passage.
"""


def format_markdown_bullets(bullets: list[str]) -> str:
    return "\n".join(f"- {bullet}" for bullet in bullets)


def format_python_list(bullets: list[str]) -> str:
    return str(bullets)


def convert_to_dataset(passage: str, out: BulletList) -> list[AlpacaDataset]:
    markdown_bullets = format_markdown_bullets(out.bullets)
    python_list_bullets = format_python_list(out.bullets)

    return [
        AlpacaDataset(
            instruction="Extract the important points from this passage as markdown bullet points.",
            input=passage,
            output=markdown_bullets,
        ),
        AlpacaDataset(
            instruction="Summarize this passage as markdown bullet points.",
            input=passage,
            output=markdown_bullets,
        ),
        AlpacaDataset(
            instruction="Extract the important points from this passage as a Python list of strings.",
            input=passage,
            output=python_list_bullets,
        ),
        AlpacaDataset(
            instruction="Return a Python list of the key points from this passage.",
            input=passage,
            output=python_list_bullets,
        ),
    ]


def main(passage: str) -> list[AlpacaDataset]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": passage},
    ]

    out = generate_structured_output(messages, BulletList, temp=0.2)
    out_augmented = augment_data(out, BulletList)

    return convert_to_dataset(passage, out) + convert_to_dataset(passage, out_augmented)


if __name__ == "__main__":
    dataset = main(
        """
The Transformer replaces recurrence and convolutions with attention mechanisms.
It improves parallelization and achieves strong machine translation performance.
It also generalizes well to other tasks such as parsing.
        """
    )

    print(len(dataset))
