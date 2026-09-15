#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama..."
  nohup ollama serve > /tmp/covalt-ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi

if ! ollama list 2>/dev/null | grep -q 'qwen2.5:1.5b'; then
  echo "Downloading qwen2.5:1.5b for Covalt..."
  ollama pull qwen2.5:1.5b
fi

export COVALT_OLLAMA_URL="${COVALT_OLLAMA_URL:-http://127.0.0.1:11434/api/chat}"
export COVALT_OLLAMA_MODEL="${COVALT_OLLAMA_MODEL:-qwen2.5:1.5b}"
export PORT="${PORT:-5000}"

exec .venv/bin/python app.py
