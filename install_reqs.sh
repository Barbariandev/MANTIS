#!/usr/bin/env bash


set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

ROOT="$(pwd)"
VENV="$ROOT/.venv"
SRC="$ROOT/timelock-src"
PY_BIN="${PY_BIN:-python3}"

echo "▶ venv : $VENV"
echo "▶ src  : $SRC"
echo "────────────────────────────────────────────────────────────"

if command -v apt-get >/dev/null 2>&1; then
  sudo rm -rf /var/lib/apt/lists/* || true
  sudo apt-get clean -qq           || true
fi

if [[ ! -d "$VENV" ]]; then
  "$PY_BIN" -m venv "$VENV"
  echo "✅ created .venv"
fi

source "$VENV/bin/activate"
python -m pip install -qU pip setuptools wheel

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
rustup toolchain install stable
rustup default stable
cargo install wasm-pack --force --locked

python -m pip install -qU maturin
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq || true
  sudo apt-get install -y --no-install-recommends pkg-config libssl-dev
elif command -v yum >/dev/null 2>&1;   then sudo yum install -y pkgconfig openssl-devel
elif command -v brew >/dev/null 2>&1;  then brew install pkg-config openssl@3
fi

# ────────────────────────────────────────────────────────────
# Install Node.js + npm (if not present) and pm2 globally
# ────────────────────────────────────────────────────────────

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "▶ Node.js not found – installing Node.js + npm…"
  if command -v apt-get >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
  elif command -v yum >/dev/null 2>&1; then
    curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
    sudo yum install -y nodejs
  elif command -v brew >/dev/null 2>&1; then
    brew install node
  else
    echo "❌ Could not detect package manager to install Node.js." >&2
    echo "   Please install Node.js manually and re-run this script." >&2
    exit 1
  fi
else
  echo "▶ Node.js already installed – skipping Node installation."
fi

# Ensure pm2 is available globally
if ! command -v pm2 >/dev/null 2>&1; then
  echo "▶ Installing pm2 globally with npm…"
  npm install -g pm2
else
  echo "▶ pm2 already installed – skipping npm install."
fi


if [[ ! -d "$SRC/.git" ]]; then
  git clone --depth 1 https://github.com/ideal-lab5/timelock.git "$SRC"
else
  git -C "$SRC" pull --ff-only
fi

sed -i.bak 's|ark_std::rand::rng::OsRng|ark_std::rand::rngs::OsRng|g' \
  "$SRC/wasm/src/"{py,js}.rs || true

pushd "$SRC/wasm" >/dev/null
maturin build --release --features python
WHEEL="$(realpath ../target/wheels/timelock_wasm_wrapper-*.whl)"
popd >/dev/null

pip install -U "$WHEEL"
pip install -U "$SRC/py"

cd "$ROOT"
if [[ -f requirements.txt ]]; then
  echo "▶ Installing project requirements with system pip3…"
  pip3 install -r requirements.txt
else
  echo "ℹ No requirements.txt found – skipping pip3 install."
fi

echo
echo "�� Timelock ready in .venv"
echo "   Activate with:  source .venv/bin/activate"

