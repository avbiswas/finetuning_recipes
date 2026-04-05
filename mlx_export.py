"""
Merge a LoRA adapter and export to MLX.

Usage:
    uv run mlx_export.py --adapter_path models/sft/smollm-135m_instruct_v1/final
    uv run mlx_export.py --adapter_path models/sft/smollm-135m_instruct_v1/final --q_bits 4
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "instruction_tuning"))
from merge_adapter import merge_adapter

parser = argparse.ArgumentParser(description="Merge LoRA adapter and convert to MLX.")
parser.add_argument("--adapter_path", "-a", type=str, required=True)
parser.add_argument("--q_bits", type=int, default=8, choices=[4, 8])
args = parser.parse_args()

checkpoint = os.path.basename(os.path.normpath(args.adapter_path))
parent = os.path.basename(os.path.dirname(args.adapter_path))

if parent in ("sft", "cpt", "models"):
    model_name = checkpoint
elif checkpoint == "final":
    model_name = parent
else:
    model_name = f"{parent}_{checkpoint}"

merged_path = f"models/merged/{model_name}"
mlx_path = f"models/mlx/{model_name}"

print("=== Step 1: Merge adapter ===")
merge_adapter(args.adapter_path, merged_path)

print(f"\n=== Step 2: Convert to MLX ({args.q_bits}-bit) ===")
subprocess.run(
    [sys.executable, "-m", "mlx_lm", "convert",
     "--hf-path", merged_path,
     "--mlx-path", mlx_path,
    # "--quantize", "--q-bits", str(args.q_bits)
     ],
    check=True,
)

print(f"\nDone.")
print(f"  Merged : {merged_path}")
print(f"  MLX    : {mlx_path}")
