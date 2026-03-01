import argparse
import json
import random

from bullets import main as generate_bullets
from comparison import main as generate_comparisons
from continuation import main as generate_continuation
from qa_pairs import main as generate_qa_pairs
from rephrase import main as generate_rephrase
from retrieval import main as generate_retrieval
from triplets import main as generate_triplets
from utils import save_dataset

PROB_TO_RUN_STEP = 0.25
PROB_TO_RUN_REPHRASE = 0.1

def chunk_text(
    text: str,
    chunk_size_words: int = 500,
    overlap_words: int = 100,
) -> list[str]:
    words = text.split()
    if not words:
        return []

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    step = chunk_size_words - overlap_words

    return [
        " ".join(words[idx: idx + chunk_size_words])
        for idx in range(0, len(words), step)
    ]


def load_texts_from_jsonl(path: str) -> list[str]:
    texts = []

    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            row = json.loads(raw)
            text = row.get("text", "")
            if not isinstance(text, str):
                raise ValueError(
                    f"Expected 'text' to be a string on line {line_number}"
                )

            text = text.strip()
            if text:
                texts.append(text)

    return texts


def build_comparison_input(left_chunk: str, right_chunk: str) -> str:
    return (
        f"Passage 1:\n{left_chunk}\n\n"
        f"Passage 2:\n{right_chunk}"
    )


def generate_examples_for_chunk(chunk: str):
    dataset = []

    if random.random() < PROB_TO_RUN_STEP:
        print("Generating bullets")
        dataset.extend(generate_bullets(chunk))

    if random.random() < PROB_TO_RUN_STEP:
        print("Generating qa pairs")
        dataset.extend(generate_qa_pairs(chunk))

    if random.random() < PROB_TO_RUN_REPHRASE:
        print("Generating rephrase")
        dataset.extend(generate_rephrase(chunk))

    if random.random() < PROB_TO_RUN_STEP:
        print("Generating Continuation")
        dataset.extend(generate_continuation(chunk))

    if random.random() < PROB_TO_RUN_STEP:
        print("Generating triplets")
        dataset.extend(generate_triplets(chunk))

    return dataset


def generate_cross_chunk_examples(chunks: list[str]):
    dataset = []

    if len(chunks) <= 2:
        print("Generating retrieval")
        if random.random() < PROB_TO_RUN_STEP:
            dataset.extend(generate_retrieval(chunks))

        if random.random() < PROB_TO_RUN_REPHRASE:
            left_idx, right_idx = random.sample(
                range(len(chunks)), 2)
            comparison_input = build_comparison_input(
                chunks[left_idx],
                chunks[right_idx],
            )
            print("Generating comparisons")
            dataset.extend(
                generate_comparisons(comparison_input))

    return dataset


def truncate_dataset(dataset, remaining_rows):
    if remaining_rows is None:
        return dataset
    return dataset[:remaining_rows]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", help="Path to the input JSONL file")
    parser.add_argument("output_jsonl", help="Path to the output JSONL file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of words per text chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Number of overlapping words between consecutive chunks",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop early after generating at least this many rows",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    texts = load_texts_from_jsonl(args.input_jsonl)
    print(f"Loaded {len(texts)} texts from {args.input_jsonl}")

    total_chunks = 0
    total_examples = 0
    texts = texts[110:]

    for text_idx, text in enumerate(texts, start=1):
        print(f"Processing text {text_idx}/{len(texts)}")
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        if not chunks:
            print(f"Skipping text {text_idx}: no valid chunks")
            continue

        total_chunks += len(chunks)
        print(
            f"Generated {len(chunks)} chunks for text {text_idx}. "
            f"Total chunks so far: {total_chunks}"
        )

        print(f"Generating synthetic examples for text {text_idx}")
        rows_generated_for_text = 0

        total_chunks = len(chunks)
        chunks = chunks[:int(total_chunks//2)]
        for chunk_idx, chunk in enumerate(chunks, start=1):
            print(f"Processing chunk {chunk_idx}/{len(chunks)} for text {text_idx}")
            dataset = []
            try:
                dataset = generate_examples_for_chunk(chunk)
            except:
                print("Errored out!\n\n\n")
                pass
            print(
                f"Generated {len(dataset)} rows for chunk {chunk_idx} "
                f"of text {text_idx} before truncation"
            )

            if args.max_rows is not None:
                remaining_rows = args.max_rows - total_examples
                if remaining_rows <= 0:
                    print("Reached max row limit before saving more rows. Stopping.")
                    break
                dataset = truncate_dataset(dataset, remaining_rows)
                print(
                    f"Keeping {len(dataset)} rows for chunk {chunk_idx} "
                    f"of text {text_idx} after applying max row limit"
                )

            total_examples += len(dataset)
            rows_generated_for_text += len(dataset)
            save_dataset(dataset, args.output_jsonl)
            print(f"Total rows saved so far: {total_examples}")

            if args.max_rows is not None and total_examples >= args.max_rows:
                print(f"Reached max row limit of {args.max_rows}. Stopping early.")
                break

        if args.max_rows is not None and total_examples >= args.max_rows:
            break

        dataset = generate_cross_chunk_examples(chunks)
        print(
            f"Generated {len(dataset)} cross-chunk rows for text {text_idx} "
            f"before truncation"
        )

        if args.max_rows is not None:
            remaining_rows = args.max_rows - total_examples
            if remaining_rows <= 0:
                print("Reached max row limit before saving more rows. Stopping.")
                break
            dataset = truncate_dataset(dataset, remaining_rows)
            print(
                f"Keeping {len(dataset)} cross-chunk rows for text {text_idx} "
                f"after applying max row limit"
            )

        total_examples += len(dataset)
        rows_generated_for_text += len(dataset)
        save_dataset(dataset, args.output_jsonl)
        print(
            f"Generated {rows_generated_for_text} total rows for text {text_idx}"
        )
        print(f"Total rows saved so far: {total_examples}")

        if args.max_rows is not None and total_examples >= args.max_rows:
            print(f"Reached max row limit of {args.max_rows}. Stopping early.")
            break

    print(
        f"Processed {len(texts)} texts into {total_chunks} chunks "
        f"and generated {total_examples} examples."
    )


if __name__ == "__main__":
    main()
