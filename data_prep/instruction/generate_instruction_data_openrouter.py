import asyncio
import argparse
import json
from typing import List
import instructor
from pydantic import BaseModel, Field
from datasets import load_dataset

client = instructor.from_provider(
    "openrouter/google/gemini-2.5-flash-lite",
    async_client=True,
    mode=instructor.Mode.JSON
)
class QA(BaseModel):
    question: str
    answer: str

class QADataset(BaseModel):
    example: List[QA] = Field()

semaphore = asyncio.Semaphore(100)

async def generate_qa(passage):
    async with semaphore: 
        examples = await client.chat.completions.create(
            response_model=QADataset,
            messages=[{"role": "user", "content": passage}]
        )
        return [
            {
                "passage": passage,
                "question": e.question,
                "answer": e.answer
            }
            for e in examples.example
        ]

async def generate(passages):
    results_list = await asyncio.gather(*[generate_qa(passage) for passage in passages])
    results = []
    for r in results_list:
        results.extend(r)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate instruction data from a JSONL file.")
    parser.add_argument("--input_jsonl_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_jsonl_path", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--n_per_paper", type=int, required=True, help="Num examples")
    args = parser.parse_args()

    # Load the input JSONL dataset
    dataset = load_dataset("json", data_files=args.input_jsonl_path, split="train")
    texts = [item["text"] for item in dataset]

    # Function to chunk text into passages
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            if end == len(text):
                break
            start += (chunk_size - overlap)
            # Ensure we don't go past the end with overlap adjustment
            if start >= len(text) and len(chunks) > 0 and (len(text) - (start - (chunk_size - overlap))) < chunk_size/2:
                # If the last chunk is too small and there are previous chunks, merge it with the previous one
                if len(chunks) > 1:
                    chunks[-2] = chunks[-2] + chunks[-1]
                    chunks.pop()
                break
            elif start >= len(text):
                break

        return chunks[:args.n_per_paper]

    # Generate passages from the loaded texts
    all_passages = []
    for text in texts:
        all_passages.extend(chunk_text(text))
    # Run the generation process
    generated_qa_pairs = asyncio.run(generate(all_passages))

    # Save the generated QA pairs to the output JSONL file
    with open(args.output_jsonl_path, "w") as f:
        print(args.output_jsonl_path)
        for entry in generated_qa_pairs:
            f.write(json.dumps(entry) + "\n")

