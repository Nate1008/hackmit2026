#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir/web"

export PATH="$repo_dir/.conda/envs/shaper-web/bin:$PATH"
web_port="${SHAPER_WEB_PORT:-3000}"

if curl -fsS "http://127.0.0.1:$web_port" >/dev/null 2>&1; then
  echo "ShapeR frontend is already running at http://localhost:$web_port"
  exit 0
fi

exec npm run dev -- --host "${SHAPER_WEB_HOST:-0.0.0.0}" --port "$web_port"

