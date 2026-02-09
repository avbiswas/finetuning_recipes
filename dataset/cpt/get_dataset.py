import sys
import json
from pathlib import Path

directory = Path(sys.argv[1])
all_files = directory.rglob(f"*.txt")
all_files = [f for f in all_files]
num_files = len(all_files)
num_test_files = int(num_files * 0.1)

print(f"Train files: {num_files - num_test_files}, Test files: {num_test_files}")

with open("cpt_val_dataset.jsonl", "a") as f:
    for a in all_files[:num_test_files]:
        content = {"text": open(a).read()}
        f.write(json.dumps(content)+"\n")

with open("cpt_train_dataset.jsonl", "a") as f:
    for a in all_files[num_test_files:]:
        content = {"text": open(a).read()}
        f.write(json.dumps(content)+"\n")

