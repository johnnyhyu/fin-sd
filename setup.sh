#!/bin/bash
# Environment bootstrap.
#
#   chmod +x setup.sh && ./setup.sh
#
# Creates a venv and installs dependencies. Everything about how the pipeline
# runs — GPU placement, the managed vLLM server, the epoch boundary — is in
# docs/ARCHITECTURE.md.
set -e

# tmux is convenient for long runs but not required; skip it if apt is absent.
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y tmux
fi

echo "Installing uv..."
pip install --user --upgrade uv
export PATH="$HOME/.local/bin:$PATH"

echo "Creating virtual environment..."
uv venv venv
source venv/bin/activate

echo "Installing dependencies..."
# requirements.txt is the dependency spec. For a byte-identical environment,
# use requirements.lock.txt instead — the resolved set the paper's runs used.
uv pip install -r requirements.txt

cat <<'MSG'

Setup complete.

  1. Activate:   source venv/bin/activate
  2. Configure:  cp .env.example .env    # then fill in OPENROUTER_API_KEY
  3. Train:      python launch.py --config configs/finsd.env

The pipeline OWNS the vLLM process — it starts the server and restarts it on a
freshly merged checkpoint each epoch. Do NOT run `vllm serve` yourself; it will
collide on the port.

  docs/RUNNING.md       train an arm, serve a checkpoint, evaluate
  docs/ARCHITECTURE.md  serving, GPU placement, the epoch boundary
  docs/CONFIG.md        every knob
MSG
