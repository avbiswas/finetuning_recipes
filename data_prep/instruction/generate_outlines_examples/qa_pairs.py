import json

from pydantic_core.core_schema import model_ser_schema
from pydantic import BaseModel, Field
from utils import AlpacaDataset, save_dataset
from model import generate_structured_output

class QA(BaseModel):
    question: str
    answer: str

class QAList(BaseModel):
    qa_pairs: list[QA] = Field(..., min_length=1,
                               max_length=3)


system_prompt = """
Given this passage of text, generate a list of important question answer pairs.
    """


def format_qa_pairs_markdown(qa_pairs):
    sections = []
    for idx, qa_pair in enumerate(qa_pairs, start=1):
        sections.append(
            f"### Q{idx}\n"
            f"**Question:** {qa_pair.question}\n\n"
            f"**Answer:** {qa_pair.answer}"
        )
    return "\n\n".join(sections)


def format_questions_markdown(qa_pairs):
    return "\n".join(
        f"{idx}. {qa_pair.question}"
        for idx, qa_pair in enumerate(qa_pairs, start=1)
    )


def format_facts_markdown(qa_pairs):
    return "\n".join(
        f"- {qa_pair.answer}"
        for qa_pair in qa_pairs
    )


def format_single_qa_markdown(qa_pair):
    return (
        f"**Question:** {qa_pair.question}\n\n"
        f"**Answer:** {qa_pair.answer}"
    )

def convert_to_json_dataset(passage, out):
    dataset = []

    dataset.append(
        AlpacaDataset(
            instruction=system_prompt + "Generate as a list of json containing 'question' and 'answer' keys",
            input=passage,
            output=json.dumps(
                [qa_pair.model_dump() for qa_pair in out.qa_pairs],
                ensure_ascii=False,
            )
        )
    )

    dataset.append(
        AlpacaDataset(
            instruction="Generate a list of questions from this passage. Return a JSON array of strings.",
            input=passage,
            output=json.dumps(
                [qa_pair.question for qa_pair in out.qa_pairs],
                ensure_ascii=False,
            )
        )
    )

    dataset.append(
        AlpacaDataset(
            instruction="List the important questions answered by this passage. Return a JSON array of strings.",
            input=passage,
            output=json.dumps(
                [qa_pair.question for qa_pair in out.qa_pairs],
                ensure_ascii=False,
            )
        )
    )

    dataset.append(
        AlpacaDataset(
            instruction="Generate some facts from this passage. Return a JSON array of strings.",
            input=passage,
            output=json.dumps(
                [qa_pair.answer for qa_pair in out.qa_pairs],
                ensure_ascii=False,
            )
        )
    )

    for qa_pair in out.qa_pairs:
        dataset.append(
            AlpacaDataset(
                instruction="Generate one question and it's corresponding answer from this passage. Return answer as a json of question and answer",
                input=passage,
                output=str(qa_pair.model_dump_json())
            )
        )
        dataset.append(
            AlpacaDataset(
                instruction="Generate a question from this passage",
                input=passage,
                output=str(qa_pair.question)
            )
        )
        dataset.append(
            AlpacaDataset(
                instruction="Generate an important fact or piece of information from this passage",
                input=passage,
                output=str(qa_pair.answer)
            )
        )
        dataset.append(
            AlpacaDataset(
                instruction="Answer the user's question given the provided passage",
                input=f"Passage: {passage}\n\nQuestion: {qa_pair.question}\nWhat is the answer?",
                output=str(qa_pair.answer)
            )
        )
        dataset.append(
            AlpacaDataset(
                instruction=f"Given the provided passage, answer the user's question. Passage: {passage}",
                input=f"{qa_pair.question}",
                output=str(qa_pair.answer)
            )
        )
    return dataset


def convert_to_dataset(passage, out):
    qa_pairs_markdown = format_qa_pairs_markdown(out.qa_pairs)
    questions_markdown = format_questions_markdown(out.qa_pairs)
    facts_markdown = format_facts_markdown(out.qa_pairs)
    dataset = []

    dataset.append(
        AlpacaDataset(
            instruction=system_prompt,
            input=passage,
            output=qa_pairs_markdown
        )
    )

    dataset.append(
        AlpacaDataset(
            instruction="Generate a set of questions from this passage in markdown format.",
            input=passage,
            output=questions_markdown
        )
    )

    dataset.append(
        AlpacaDataset(
            instruction="List the important questions answered by this passage using markdown.",
            input=passage,
            output=questions_markdown
        )
    )

    dataset.append(
        AlpacaDataset(
            instruction="Generate some important facts from this passage in markdown bullet points.",
            input=passage,
            output=facts_markdown
        )
    )

    for qa_pair in out.qa_pairs:
        dataset.append(
            AlpacaDataset(
                instruction="Generate one question and its corresponding answer from this passage in markdown format.",
                input=passage,
                output=format_single_qa_markdown(qa_pair)
            )
        )
        dataset.append(
            AlpacaDataset(
                instruction="Generate a question from this passage",
                input=passage,
                output=str(qa_pair.question)
            )
        )
        dataset.append(
            AlpacaDataset(
                instruction="Generate an important fact or piece of information from this passage",
                input=passage,
                output=str(qa_pair.answer)
            )
        )
        dataset.append(
            AlpacaDataset(
                instruction="Answer the user's question given the provided passage",
                input=f"Passage: {passage}\n\nQuestion: {qa_pair.question}\nWhat is the answer?",
                output=str(qa_pair.answer)
            )
        )
        dataset.append(
            AlpacaDataset(
                instruction=f"Given the provided passage, answer the user's question. Passage: {passage}",
                input=f"{qa_pair.question}",
                output=str(qa_pair.answer)
            )
        )
    return dataset


def main(passage: str):
    messages = [
        {
            "role": "system", "content": system_prompt
        },
        {
            "role": "user", "content": passage 
        }
    ]
    
    out = generate_structured_output(messages, QAList)
    
    dataset = (
        convert_to_dataset(passage, out)
        + convert_to_json_dataset(passage, out)
    )

    return dataset

if __name__ == "__main__":

    dataset = main("""
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
             """)

    print(len(dataset))
