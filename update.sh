#!/usr/bin/env bash
# Bracket self-updater (Linux / macOS / WSL2).
#
# Pulls the latest changes from the configured git remote, re-runs the
# installer (idempotent — picks up new Python/Node deps and rebuilds the
# frontend), and re-launches the server.
#
# Triggered by the React UI via POST /api/update/apply, which spawns this
# script detached from the dying API process. Output is appended to
# runs/update.log so the user can tail it after the restart.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/runs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/update.log"

log() {
    local stamp
    stamp=$(date '+%Y-%m-%d %H:%M:%S')
    printf '[%s] %s\n' "$stamp" "$*" | tee -a "$LOG_PATH"
}

log "==> Bracket update started"

# Give the API a moment to flush its response before we yank it.
sleep 1

# Stop any running bracket-serve process listening on the configured port.
PORT="${BRACKET_PORT:-8000}"
if command -v lsof >/dev/null 2>&1; then
    PIDS=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        log "Stopping PIDs holding port $PORT: $PIDS"
        kill $PIDS 2>/dev/null || true
        sleep 2
        # Hard-kill anything still alive.
        for pid in $PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
fi

# Pull latest.
log "==> git fetch + git pull"
if ! git fetch --tags >>"$LOG_PATH" 2>&1; then
    log "git fetch failed (non-fatal — continuing with cached refs)"
fi
if ! git pull --ff-only >>"$LOG_PATH" 2>&1; then
    log "[X] git pull failed. Aborting — your tree may have local changes."
    exit 1
fi
log "==> git submodule update --init --recursive"
if ! git submodule update --init --recursive >>"$LOG_PATH" 2>&1; then
    log "[X] git submodule update failed."
    exit 1
fi

# Re-run the installer to pick up new deps and rebuild the frontend.
log "==> Re-running install.sh"
if ! bash "$REPO_ROOT/install.sh" >>"$LOG_PATH" 2>&1; then
    log "[X] install.sh failed. Server NOT relaunched. Check the log above."
    exit 1
fi

# Relaunch — detach so this updater can exit cleanly.
log "==> Relaunching server"
LAUNCH="$REPO_ROOT/launch.sh"
if [ ! -f "$LAUNCH" ]; then
    log "[X] launch.sh missing — cannot relaunch. Run it manually."
    exit 1
fi
chmod +x "$LAUNCH" 2>/dev/null || true
nohup bash "$LAUNCH" >>"$LOG_PATH" 2>&1 &
disown || true
log "==> Update complete. New server launching (PID $!)."
