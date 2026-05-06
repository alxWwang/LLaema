#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "Error: .venv not found. Run 'python -m venv .venv && .venv/bin/pip install -r requirements.txt' first." >&2
  exit 1
fi

PYTHONPATH="$SCRIPT_DIR" \
exec .venv/bin/python app.py "$@"
