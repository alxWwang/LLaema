#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  echo "Usage: $0 <target> [args...]"
  echo ""
  echo "Targets:"
  echo "  agent    Run app.py"
  echo "  context  Run context/main.py"
  exit 1
}

check_venv() {
  if [ ! -d ".venv" ]; then
    echo "Error: .venv not found. Run 'python -m venv .venv && .venv/bin/pip install -r requirements.txt' first." >&2
    exit 1
  fi
}

[ $# -lt 1 ] && usage

TARGET="$1"
shift

case "$TARGET" in
  agent)
    check_venv
    PYTHONPATH="$SCRIPT_DIR" exec .venv/bin/python app.py "$@"
    ;;
  context)
    check_venv
    PYTHONPATH="$SCRIPT_DIR" exec .venv/bin/python -m context.main "$@"
    ;;
  *)
    echo "Error: unknown target '$TARGET'" >&2
    usage
    ;;
esac
