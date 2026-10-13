#!/usr/bin/env bash
# One command: start the database, apply migrations, open the control panel.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env. Open it, paste your Alpaca keys (and any model keys), then run ./panel.sh again."
  exit 1
fi

docker compose up -d --wait postgres
CONFIG="${1:-config/local.toml}"
python -m engine.service migrate
(sleep 2 && open "http://127.0.0.1:8765" 2>/dev/null || true) &
exec python -m engine.service panel --config "$CONFIG"
