#!/usr/bin/env bash
# Bracket launcher (Linux / macOS / WSL2).
#
# Activates the repo venv, exports any defaults from .env, and starts the
# unified server (FastAPI + static frontend) on http://127.0.0.1:8000.
#
# Flags pass through to bracket-serve.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "✗ .venv not found. Run ./install.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Load .env into the environment, ignoring comments and blank lines.
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . <(grep -E '^[A-Z_]+=' "$REPO_ROOT/.env")
    set +a
fi

exec bracket-serve "$@"
