#!/usr/bin/env bash

# Autoupdate script for MANTIS validator
#
# This script periodically checks the GitHub repository for new commits on the
# specified branch, pulls updates, and restarts the pm2 process named
# "validator" if it is already running. If the process is not found, it will
# start it using the provided wallet.name and wallet.hotkey parameters, which
# can be supplied via CLI flags or loaded from a .env file.
#
# Usage examples:
#   ./autoupdate.sh                         # Uses .env for wallet vars
#   ./autoupdate.sh -w mywallet -k myhotkey # CLI overrides .env values
#   ./autoupdate.sh -i 120                 # Check every 120 seconds
#
# Flags:
#   -w|--wallet.name   The wallet name to use when starting validator
#   -k|--wallet.hotkey The wallet hotkey to use when starting validator
#   -b|--branch        Git branch to track (default: main)
#   -i|--interval      Seconds between update checks (default: 60)

set -euo pipefail

##############################
# Helper functions
##############################

log() {
  # Timestamped logging helper
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

print_usage() {
  cat <<EOF
Usage: $0 [options]
Options:
  -w, --wallet.name   <wallet_name>
  -k, --wallet.hotkey <wallet_hotkey>
  -b, --branch        <branch>       (default: main)
  -i, --interval      <seconds>      (default: 60)
EOF
}

##############################
# Parse CLI arguments
##############################

BRANCH="main"
INTERVAL=60
WALLET_NAME=""
WALLET_HOTKEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--wallet.name)
      WALLET_NAME="$2"; shift 2;;
    -k|--wallet.hotkey)
      WALLET_HOTKEY="$2"; shift 2;;
    -b|--branch)
      BRANCH="$2"; shift 2;;
    -i|--interval)
      INTERVAL="$2"; shift 2;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown parameter: $1" >&2
      print_usage; exit 1;;
  esac
done

##############################
# Load .env if present
##############################

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi

# Environment variables take precedence over .env, CLI overrides all.
: "${WALLET_NAME:=${WALLET_NAME:-${wallet_name:-}}}"
: "${WALLET_HOTKEY:=${WALLET_HOTKEY:-${wallet_hotkey:-}}}"

if [[ -z "$WALLET_NAME" || -z "$WALLET_HOTKEY" ]]; then
  log "wallet.name and wallet.hotkey must be provided via CLI or .env"
  exit 1
fi

# Ensure we are running from the repository root
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

##############################
# Core logic
##############################

check_for_updates() {
  local remote_commit local_commit

  # Fetch remote changes but don't merge yet
  git fetch origin "$BRANCH" &>/dev/null || { log "Failed to fetch remote repository"; return 1; }

  remote_commit=$(git rev-parse "origin/$BRANCH")
  local_commit=$(git rev-parse "HEAD")

  if [[ "$remote_commit" != "$local_commit" ]]; then
    log "New commit detected. Pulling changes..."
    git pull --rebase origin "$BRANCH" || { log "git pull failed"; return 1; }
    return 0  # Updates applied
  else
    return 1  # No updates
  fi
}

restart_or_start_validator() {
  if pm2 list | grep -q "validator"; then
    log "Restarting existing pm2 process 'validator'..."
    pm2 restart validator
  else
    log "Starting pm2 process 'validator'..."
    pm2 start validator.py -- --wallet.name "$WALLET_NAME" --wallet.hotkey "$WALLET_HOTKEY"
  fi

  # Begin or refresh log streaming
  stream_validator_logs
}

ensure_validator_running() {
  if ! pm2 list | grep -q "validator"; then
    restart_or_start_validator
  else
    # Validator already up: ensure logs streaming is active
    if [[ -z "$LOG_PID" || ! -e /proc/$LOG_PID ]]; then
      stream_validator_logs
    fi
  fi
}

# ────────────────────────────────────────────────────────────
# Log streaming
# ────────────────────────────────────────────────────────────

LOG_PID=""

stream_validator_logs() {
  if [[ -n "$LOG_PID" && -e /proc/$LOG_PID ]]; then
    kill "$LOG_PID" || true
  fi

  pm2 logs validator --lines 20 --raw &
  LOG_PID=$!
}

# Clean up background log stream on exit
trap '[[ -n "$LOG_PID" && -e /proc/$LOG_PID ]] && kill "$LOG_PID"' EXIT



log "Starting autoupdate loop (branch=$BRANCH, interval=${INTERVAL}s)"
ensure_validator_running  # Ensure validator is up on startup

while true; do
  if check_for_updates; then
    restart_or_start_validator
  else
    ensure_validator_running
  fi
  sleep "$INTERVAL"
done 
