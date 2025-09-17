#!/usr/bin/env bash
# install_reqs.sh
# Sets up a Python venv at .venv and installs the 'timelock' Python package.
# Strategy:
#   1) Try to install prebuilt wheels from PyPI (works on Python 3.10).
#   2) If that fails, build from source (Rust + maturin) for the active interpreter.
#
# Env vars:
#   PY_BIN        : Python interpreter to use (default: python3.10 if present, else python3)
#   SRC           : Where to clone timelock sources (default: ./timelock-src)
#   INSTALL_NODE  : If "1", also install Node.js 20 + pm2 (default: 0/disabled)

set -Eeuo pipefail
IFS=$'\n\t'
export DEBIAN_FRONTEND=noninteractive

step() { echo -e "\n\033[1;36m▶ $*\033[0m"; }
ok()   { echo -e "\033[1;32m✔ $*\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $*\033[0m"; }
die()  { echo -e "\033[1;31m✖ $*\033[0m" >&2; exit 1; }
trap 'die "Error on or near line $LINENO. Aborting."' ERR

ROOT="$(pwd)"
VENV="$ROOT/.venv"
SRC="${SRC:-$ROOT/timelock-src}"

# Choose interpreter
if [[ -n "${PY_BIN:-}" ]]; then
  : # use override
elif command -v python3.10 >/dev/null 2>&1; then
  PY_BIN="python3.10"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
else
  die "Could not find a suitable Python interpreter (need python3)."
fi

SUDO=""
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  SUDO="sudo"
fi

step "Using interpreter: $($PY_BIN --version 2>&1 | tr -d '\n')"

# Create (or reuse) venv at .venv
if [[ ! -d "$VENV" ]]; then
  step "Creating virtualenv at $VENV"
  "$PY_BIN" -m venv "$VENV"
  ok "Created $VENV"
else
  warn "$VENV already exists – reusing it"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PY="$VENV/bin/python"
PIP="$PY -m pip"

# Upgrade pip tooling
$PY -m pip install -qU pip setuptools wheel

# Optional: Node.js + pm2
if [[ "${INSTALL_NODE:-0}" == "1" ]]; then
  step "Checking Node.js + pm2"
  if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      curl -fsSL https://deb.nodesource.com/setup_20.x | $SUDO -E bash -
      $SUDO apt-get install -y nodejs
    elif command -v dnf >/dev/null 2>&1; then
      curl -fsSL https://rpm.nodesource.com/setup_20.x | $SUDO bash -
      $SUDO dnf install -y nodejs
    elif command -v brew >/dev/null 2>&1; then
      brew install node
    else
      die "No supported package manager found to install Node.js."
    fi
  else
    ok "Node.js already present"
  fi
  if ! command -v pm2 >/dev/null 2>&1; then
    npm install -g pm2
    ok "Installed pm2"
  else
    ok "pm2 already present"
  fi
else
  warn "Skipping Node.js + pm2 install (set INSTALL_NODE=1 to enable)"
fi

# Fast path: try prebuilt wheels from PyPI (works on Python 3.10)
step "Attempting timelock install from PyPI (prebuilt wheels)"
PREBUILT=0
if "$PY" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("timelock") else 1)' >/dev/null 2>&1; then
  ok "timelock already installed in venv"
  PREBUILT=1
else
  if $PY -m pip install -q --no-input timelock; then
    if "$PY" -c 'import timelock' >/dev/null 2>&1; then
      ok "Installed timelock from PyPI"
      PREBUILT=1
    else
      warn "timelock import failed after PyPI install; will build from source"
      PREBUILT=0
    fi
  else
    warn "PyPI install did not succeed; will build from source"
    PREBUILT=0
  fi
fi

# Build from source (Rust + maturin) if needed
if [[ "$PREBUILT" -eq 0 ]]; then
  step "Installing system build dependencies"
  if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -qq || true
    $SUDO apt-get install -y --no-install-recommends \
      build-essential pkg-config libssl-dev ca-certificates git curl
    # Ensure matching Python headers are present
    pyver="$($PY -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    if ! dpkg -s "python${pyver}-dev" >/dev/null 2>&1; then
      $SUDO apt-get install -y "python${pyver}-dev" || $SUDO apt-get install -y python3-dev || true
    fi
  elif command -v dnf >/dev/null 2>&1; then
    $SUDO dnf install -y gcc gcc-c++ make pkgconf-pkg-config openssl-devel git curl python3-devel
  elif command -v brew >/dev/null 2>&1; then
    brew install pkg-config openssl@3 git curl
  fi

  step "Ensuring Rust toolchain"
  if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  fi
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
  rustup toolchain install -q stable
  rustup default -q stable

  step "Cloning/updating timelock sources"
  if [[ ! -d "$SRC/.git" ]]; then
    git clone --depth 1 https://github.com/ideal-lab5/timelock.git "$SRC"
  else
    git -C "$SRC" pull --ff-only
  fi

  # Backward-compat fix for ark_std rename (no-op if not needed)
  if [[ -d "$SRC/wasm/src" ]]; then
    sed -i.bak 's|ark_std::rand::rng::OsRng|ark_std::rand::rngs::OsRng|g' "$SRC/wasm/src/"{py,js}.rs || true
  fi

  step "Building timelock_wasm_wrapper for this interpreter"
  $PY -m pip install -qU maturin
  pushd "$SRC/wasm" >/dev/null
  $PY -m maturin build --release --features "python" --interpreter "$PY"
  popd >/dev/null

  # Find the built wheel (workspace vs crate target dir)
  WHEEL_PATH="$(ls -1 "$SRC"/target/wheels/timelock_wasm_wrapper-*.whl 2>/dev/null | head -n1 || true)"
  if [[ -z "${WHEEL_PATH:-}" ]]; then
    WHEEL_PATH="$(ls -1 "$SRC"/wasm/target/wheels/timelock_wasm_wrapper-*.whl 2>/dev/null | head -n1 || true)"
  fi
  [[ -n "${WHEEL_PATH:-}" ]] || die "Failed to find built wheel in target/wheels"

  $PY -m pip install -U "$WHEEL_PATH"

  step "Installing Python bindings"
  $PY -m pip install -U "$SRC/py"

  ok "Built and installed timelock from source"
fi

# Project requirements (install into the venv)
if [[ -f "$ROOT/requirements.txt" ]]; then
  step "Installing project requirements into .venv"
  $PY -m pip install -r "$ROOT/requirements.txt"
else
  warn "No requirements.txt found – skipping"
fi

echo
ok "Timelock ready in $VENV"
echo "Activate with:  source \"$VENV/bin/activate\""



