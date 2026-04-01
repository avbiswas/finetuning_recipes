import argparse
import random
from functools import lru_cache

import outlines
from text_albumentations import OutlinesModel, run_augmentation, run_batch_augmentation
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import (
    PROB_TO_RUN_REPHRASE,
    PROB_TO_RUN_STEP,
    bullet_augmentation,
    chunk_text,
    comparison_augmentation,
    continuation_augmentation,
    load_texts_from_jsonl,
    qa_pair_augmentation,
    rephrase_augmentation,
    retrieval_augmentation,
    save_dataset,
    try_generate,
    triplet_augmentation,
    truncate_dataset,
)

MODEL_NAME = "google/gemma-3-1b-it"


def batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


@lru_cache(maxsize=1)
def get_batch_runtime():
    print(f"loading_transformers_model model_name={MODEL_NAME}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    print("loading_tokenizer")
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("building_outlines_model")
    model = outlines.from_transformers(hf_model, hf_tokenizer)
    return OutlinesModel(model, max_tokens_parameter="max_new_tokens")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", help="Path to the input JSONL file")
    parser.add_argument("output_jsonl", help="Path to the output JSONL file")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this text index in the input JSONL",
    )
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
        "--batch-size",
        type=int,
        default=8,
        help="Number of passages to decode together for batch-compatible tasks",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop early after generating at least this many rows",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for task sampling",
    )
    return parser.parse_args()


def generate_examples_for_chunks_batch(chunks: list[str], batch_size: int):
    runtime = get_batch_runtime()
    dataset = []

    single_chunk_tasks = [
        ("bullets", bullet_augmentation, PROB_TO_RUN_STEP),
        ("qa_pairs", qa_pair_augmentation, PROB_TO_RUN_STEP),
        ("rephrase", rephrase_augmentation, PROB_TO_RUN_REPHRASE),
        ("continuation", continuation_augmentation, PROB_TO_RUN_STEP),
        ("triplets", triplet_augmentation, PROB_TO_RUN_STEP),
    ]

    for task_name, augmentation, probability in single_chunk_tasks:
        selected_chunks = [
            chunk for chunk in chunks
            if random.random() < probability
        ]
        if not selected_chunks:
            continue

        print(
            f"Running batch augmentation for {task_name} on "
            f"{len(selected_chunks)} chunks"
        )
        for chunk_batch in batched(selected_chunks, batch_size):
            dataset.extend(
                try_generate(
                    f"{task_name} batch",
                    lambda augmentation=augmentation, chunk_batch=chunk_batch: (
                        run_batch_augmentation(chunk_batch, augmentation, runtime)
                    ),
                )
            )

    return dataset


def generate_cross_chunk_examples(chunks: list[str]):
    runtime = get_batch_runtime()
    dataset = []

    if len(chunks) >= 2:
        if random.random() < PROB_TO_RUN_STEP:
            print("Generating retrieval")
            dataset.extend(
                try_generate(
                    "retrieval",
                    lambda: run_augmentation(chunks, retrieval_augmentation, runtime),
                )
            )

        if random.random() < PROB_TO_RUN_REPHRASE:
            left_idx, right_idx = random.sample(range(len(chunks)), 2)
            print("Generating comparisons")
            dataset.extend(
                try_generate(
                    "comparison",
                    lambda: run_augmentation(
                        [chunks[left_idx], chunks[right_idx]],
                        comparison_augmentation,
                        runtime,
                    ),
                )
            )

    return dataset


def main():
    args = parse_args()
    random.seed(args.seed)

    texts = load_texts_from_jsonl(args.input_jsonl)
    print(f"Loaded {len(texts)} texts from {args.input_jsonl}")
    print(f"batch_model_name={MODEL_NAME}")

    total_chunks = 0
    total_examples = 0
    texts = texts[args.start_index:]

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

        chunks = chunks[: int(len(chunks) // 2)]
        print(f"Running batch generation over {len(chunks)} chunks")

        dataset = generate_examples_for_chunks_batch(chunks, args.batch_size)
        print(
            f"Generated {len(dataset)} single-chunk rows for text {text_idx} "
            f"before truncation"
        )

        if args.max_rows is not None:
            remaining_rows = args.max_rows - total_examples
            if remaining_rows <= 0:
                print("Reached max row limit before saving more rows. Stopping.")
                break
            dataset = truncate_dataset(dataset, remaining_rows)

        total_examples += len(dataset)
        save_dataset(dataset, args.output_jsonl)
        print(f"Total rows saved so far: {total_examples}")

        if args.max_rows is not None and total_examples >= args.max_rows:
            print(f"Reached max row limit of {args.max_rows}. Stopping early.")
            break

        cross_chunk_dataset = generate_cross_chunk_examples(chunks)
        print(
            f"Generated {len(cross_chunk_dataset)} cross-chunk rows for text "
            f"{text_idx} before truncation"
        )

        if args.max_rows is not None:
            remaining_rows = args.max_rows - total_examples
            if remaining_rows <= 0:
                print("Reached max row limit before saving more rows. Stopping.")
                break
            cross_chunk_dataset = truncate_dataset(
                cross_chunk_dataset,
                remaining_rows,
            )

        total_examples += len(cross_chunk_dataset)
        save_dataset(cross_chunk_dataset, args.output_jsonl)
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
