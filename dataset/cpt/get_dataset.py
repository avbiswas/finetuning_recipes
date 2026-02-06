import sys
import json
from pathlib import Path

directory = Path(sys.argv[1])
all_files = directory.rglob(f"*.txt")

with open("cpt_dataset.jsonl", "a") as f:
    for a in all_files:
        content = {"text": open(a).read()}
        f.write(json.dumps(content)+"\n")
