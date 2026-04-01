import argparse
import asyncio
import os
import random

from text_albumentations import OutlinesModel, arun_augmentation, save_dataset
from text_albumentations.tasks import (
    bullet_augmentation,
    comparison_augmentation,
    continuation_augmentation,
    qa_pair_augmentation,
    rephrase_augmentation,
    retrieval_augmentation,
    triplet_augmentation,
)

from main import (
    PROB_TO_RUN_REPHRASE,
    PROB_TO_RUN_STEP,
    chunk_text,
    load_texts_from_jsonl,
    try_generate,
    truncate_dataset,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
        "--model-name",
        type=str,
        default=os.environ.get("TEXT_ALBUMENTATIONS_MODEL", "gpt-5.4-nano"),
        help="OpenAI-compatible model name for async generation",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "openrouter"],
        default=os.environ.get("LLM_PROVIDER", "openai"),
        help="Which OpenAI-compatible provider client path to use",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL override",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI-compatible API key",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop early after generating at least this many rows",
    )
    parser.add_argument(
        "--total-concurrent-calls",
        type=int,
        default=8,
        help="Maximum concurrent async model calls",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for task sampling",
    )
    return parser.parse_args()


def resolve_client_config(args):
    if args.provider == "openrouter":
        return {
            "api_key": args.api_key or os.environ.get("OPENROUTER_API_KEY"),
            "base_url": args.base_url or OPENROUTER_BASE_URL,
        }

    return {
        "api_key": args.api_key or os.environ.get("OPENAI_API_KEY"),
        "base_url": args.base_url or os.environ.get("OPENAI_BASE_URL"),
    }


def build_async_runtime(args):
    import openai
    import outlines

    client_kwargs = resolve_client_config(args)
    client_kwargs = {
        key: value
        for key, value in client_kwargs.items()
        if value
    }

    client = openai.AsyncOpenAI(**client_kwargs)
    model = outlines.from_openai(client, args.model_name)

    return OutlinesModel(
        model,
        async_mode=True,
        total_concurrent_calls=args.total_concurrent_calls,
        max_tokens_parameter="max_completion_tokens",
    )


async def generate_examples_for_chunks_async(chunks: list[str], runtime):
    tasks = []

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
            f"Scheduling async augmentation for {task_name} on "
            f"{len(selected_chunks)} chunks"
        )
        for chunk in selected_chunks:
            tasks.append(arun_augmentation(chunk, augmentation, runtime))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    dataset = []
    for result in results:
        if isinstance(result, KeyboardInterrupt):
            raise result
        if isinstance(result, Exception):
            print(f"Skipping async augmentation: {result}")
            continue
        dataset.extend(result)
    return dataset


async def generate_cross_chunk_examples_async(chunks: list[str], runtime):
    tasks = []

    if len(chunks) >= 2:
        if random.random() < PROB_TO_RUN_STEP:
            print("Scheduling async retrieval")
            tasks.append(arun_augmentation(chunks, retrieval_augmentation, runtime))

        if random.random() < PROB_TO_RUN_REPHRASE:
            left_idx, right_idx = random.sample(range(len(chunks)), 2)
            print("Scheduling async comparison")
            tasks.append(
                arun_augmentation(
                    [chunks[left_idx], chunks[right_idx]],
                    comparison_augmentation,
                    runtime,
                )
            )

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    dataset = []
    for result in results:
        if isinstance(result, KeyboardInterrupt):
            raise result
        if isinstance(result, Exception):
            print(f"Skipping async augmentation: {result}")
            continue
        dataset.extend(result)
    return dataset


async def amain():
    args = parse_args()
    random.seed(args.seed)

    runtime = build_async_runtime(args)

    texts = load_texts_from_jsonl(args.input_jsonl)
    print(f"Loaded {len(texts)} texts from {args.input_jsonl}")
    print(f"provider={args.provider}")
    print(f"model_name={args.model_name}")
    print(f"base_url={resolve_client_config(args)['base_url']}")
    print(f"total_concurrent_calls={args.total_concurrent_calls}")

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
        chunks = chunks[: int(len(chunks) // 2)]

        dataset = await generate_examples_for_chunks_async(chunks, runtime)
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

        cross_chunk_dataset = await generate_cross_chunk_examples_async(
            chunks,
            runtime,
        )
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
    asyncio.run(amain())
