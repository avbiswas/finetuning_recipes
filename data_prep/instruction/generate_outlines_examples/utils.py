from pydantic import BaseModel, Field

class AlpacaDataset(BaseModel):
    instruction: str
    input: str
    output: str

def save_dataset(dataset: list[AlpacaDataset],
                 filename):
    if not filename.endswith(".jsonl"):
        filename = filename + ".jsonl"
    with open(filename, "a") as f:
        for d in dataset:
            f.write(d.model_dump_json()+"\n")
    
    print(f"Saved {len(dataset)} rows in {filename}")

if __name__ == "__main__":
    save_dataset(
        [
    AlpacaDataset(instruction="1", input="2", output="3"),
    AlpacaDataset(instruction="3", input="2", output="3"),
        ],
        "test.jsonl"
    )
