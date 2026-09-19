#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

export HF_HOME="${HF_HOME:-$repo_dir/.cache/huggingface}"
backend_host="${SHAPER_HOST:-0.0.0.0}"
backend_port="${SHAPER_PORT:-8000}"

if curl -fsS "http://127.0.0.1:$backend_port/api/health" >/dev/null 2>&1; then
  echo "ShapeR backend is already running at http://localhost:$backend_port"
  exit 0
fi

exec "$repo_dir/.conda/envs/shaper/bin/uvicorn" backend.app:app \
  --host "$backend_host" \
  --port "$backend_port"

