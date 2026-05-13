#!/usr/bin/env bash

set -euo pipefail

CUDA_VERSION="12.8"
CUDA_HOME_PATH="/usr/local/cuda-${CUDA_VERSION}"
PROFILE_SNIPPET="$HOME/.cuda-env"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "Checking NVIDIA driver..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install/configure the NVIDIA driver first."
  exit 1
fi
nvidia-smi

echo "Checking PyTorch CUDA visibility..."
if command -v uv >/dev/null 2>&1; then
  uv run python - <<'PY'
import os
import torch

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("CUDA_HOME", os.environ.get("CUDA_HOME"))

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA. Fix driver/PyTorch install before continuing.")
PY
else
  echo "uv not found. Skipping PyTorch CUDA check."
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc already installed:"
  nvcc --version
else
  echo "nvcc not found. Installing CUDA toolkit ${CUDA_VERSION}..."
  $SUDO apt-get update -y
  $SUDO apt-get install -y "linux-headers-$(uname -r)" build-essential
  $SUDO apt-get install -y cuda-toolkit-12-8
fi

if [ ! -d "$CUDA_HOME_PATH" ] && [ -d /usr/local/cuda ]; then
  CUDA_HOME_PATH="/usr/local/cuda"
fi

if [ ! -d "$CUDA_HOME_PATH" ]; then
  echo "CUDA toolkit directory not found at /usr/local/cuda-${CUDA_VERSION} or /usr/local/cuda."
  exit 1
fi

cat > "$PROFILE_SNIPPET" <<EOF
export CUDA_HOME=$CUDA_HOME_PATH
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
EOF

if ! grep -qs "source $PROFILE_SNIPPET" "$HOME/.bashrc"; then
  echo "source $PROFILE_SNIPPET" >> "$HOME/.bashrc"
fi

if [ -f "$HOME/.zshrc" ] && ! grep -qs "source $PROFILE_SNIPPET" "$HOME/.zshrc"; then
  echo "source $PROFILE_SNIPPET" >> "$HOME/.zshrc"
fi

# shellcheck disable=SC1090
source "$PROFILE_SNIPPET"

echo "CUDA environment configured:"
echo "CUDA_HOME=$CUDA_HOME"
command -v nvcc
nvcc --version

if command -v uv >/dev/null 2>&1; then
  uv run python - <<'PY'
import os
import torch

print("final torch cuda available", torch.cuda.is_available())
print("final torch cuda", torch.version.cuda)
print("final device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("final CUDA_HOME", os.environ.get("CUDA_HOME"))
PY
fi

