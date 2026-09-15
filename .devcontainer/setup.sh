#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama in the Codespace..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

mkdir -p data/videos data/thumbs data/images
printf '\nCovalt dependencies are ready.\n'
