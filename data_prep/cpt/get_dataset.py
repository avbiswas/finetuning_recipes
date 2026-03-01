import sys
import json
import re
import random
from pathlib import Path

random.seed(42)


def process_text(text, filename):
    ref_matches = list(re.finditer(r"(?i)\breferences\b", text))
    if not ref_matches:
        print(f"{filename.name}: Removed 0.00%")
        return text

    last_ref = ref_matches[-1]
    ref_start = last_ref.start()
    total_len = len(text)

    if ref_start > 0.7 * total_len:
        thrown_out_len = total_len - ref_start
        if thrown_out_len <= 0.3 * total_len:
            print(f"{filename.name}: Removed {(thrown_out_len / total_len) * 100:.2f}%")
            return text[:ref_start]
    else:
        app_match = re.search(r"(?i)\bappendix\b", text[ref_start:])
        if app_match:
            app_start = ref_start + app_match.start()
            thrown_out_len = app_start - ref_start
            if thrown_out_len <= 0.3 * total_len:
                print(
                    f"{filename.name}: Removed {(thrown_out_len / total_len) * 100:.2f}%"
                )
                return text[:ref_start] + text[app_start:]

    print(f"{filename.name}: Removed 0.00%")
    return text


directory = Path(sys.argv[1])
all_files = directory.rglob(f"*.txt")
all_files = [f for f in all_files]
num_files = len(all_files)
random.shuffle(all_files)
num_test_files = 50

print(f"Train files: {num_files - num_test_files}, Test files: {num_test_files}")

with open(f"cpt_val_dataset_{num_test_files}.jsonl", "a") as f:
    for a in all_files[:num_test_files]:
        content = {"text": process_text(open(a).read(), a)}
        f.write(json.dumps(content) + "\n")

with open(f"cpt_train_dataset_{num_files - num_test_files}.jsonl", "a") as f:
    for a in all_files[num_test_files:]:
        content = {"text": process_text(open(a).read(), a)}
        f.write(json.dumps(content) + "\n")
