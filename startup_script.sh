#!/bin/bash

# Stop on error
set -e

echo "🚀 Starting setup script for Ubuntu..."

# 1. Install System Packages (tmux, vim, git, curl)
echo "📦 Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y tmux vim curl git

# 2. Install uv
if ! command -v uv &> /dev/null; then
    echo "⚡ Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Ensure uv is in path for this session if possible, though mostly for next session
    # standard install puts it in ~/.cargo/bin
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ uv is already installed."
fi

# 3. Fix "missing or unsuitable terminal: xterm-ghostty"
# This happens when connecting from Ghostty terminal to a server without its terminfo.
if [[ "$TERM" == "xterm-ghostty" ]]; then
    echo "👻 Ghostty terminal detected. Applying compatibility fix..."
    echo "export TERM=xterm-256color" >> ~/.bashrc
    export TERM=xterm-256color
    echo "✅ Added 'export TERM=xterm-256color' to ~/.bashrc"
fi

echo "🎉 Setup complete!"
echo "Run 'source \$HOME/.cargo/env' to activate uv in this shell."
