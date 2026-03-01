import json

from pydantic import BaseModel, Field

from model import augment_data, generate_structured_output
from utils import AlpacaDataset


class Triplet(BaseModel):
    subject: str = Field(..., max_length=120)
    relation: str = Field(..., max_length=120)
    object: str = Field(..., max_length=160)


class TripletList(BaseModel):
    triplets: list[Triplet] = Field(..., min_length=1, max_length=2)


SYSTEM_PROMPT = """
Given this passage, extract a small set of knowledge graph triplets.
Each triplet must be directly supported by the passage.
Use short, clear phrases for the subject, relation, and object.
Do not invent facts not present in the passage.
"""


def format_triplets_markdown(triplets: list[Triplet]) -> str:
    return "\n".join(
        f"- ({triplet.subject}, {triplet.relation}, {triplet.object})"
        for triplet in triplets
    )


def format_triplets_json(triplets: list[Triplet]) -> str:
    return json.dumps(
        [triplet.model_dump() for triplet in triplets],
        ensure_ascii=False,
    )


def convert_to_dataset(passage: str, out: TripletList) -> list[AlpacaDataset]:
    markdown_triplets = format_triplets_markdown(out.triplets)
    json_triplets = format_triplets_json(out.triplets)

    return [
        AlpacaDataset(
            instruction="Extract knowledge graph triplets from this passage in markdown format.",
            input=passage,
            output=markdown_triplets,
        ),
        AlpacaDataset(
            instruction="List the subject-relation-object triplets from this passage as markdown bullet points.",
            input=passage,
            output=markdown_triplets,
        ),
        AlpacaDataset(
            instruction="Extract knowledge graph triplets from this passage and return them as JSON.",
            input=passage,
            output=json_triplets,
        ),
        AlpacaDataset(
            instruction="Return a JSON array of subject-relation-object triplets supported by this passage.",
            input=passage,
            output=json_triplets,
        ),
    ]


def main(passage: str) -> list[AlpacaDataset]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": passage},
    ]

    out = generate_structured_output(messages, TripletList, temp=0.2)
    return convert_to_dataset(passage, out)


if __name__ == "__main__":
    dataset = main(
        """
The Transformer uses attention mechanisms.
The model achieves strong results on machine translation tasks.
The architecture removes recurrence and convolutions.
        """
    )

    print(len(dataset))
