#!/usr/bin/env bash
# install_reqs.sh
# Sets up a Python venv at .venv and installs project deps + timelock.
# Timelock is always built from the pinned ideal-lab5/timelock revision.
# PyPI `timelock==0.0.1.dev0` + `timelock-wasm-wrapper==0.0.2` is the old
# 372-byte stack and cannot decrypt live (356-byte) miner payloads.
#
# Env vars:
#   PY_BIN        : Python interpreter to use (default: auto-detect, prefers 3.10)
#   SRC           : Where to clone timelock sources (default: ./timelock-src)
#   TIMELOCK_GIT  : timelock repo URL
#   TIMELOCK_REV  : pinned commit (must ship wasm wrapper 0.3.0)
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
TIMELOCK_GIT="${TIMELOCK_GIT:-https://github.com/ideal-lab5/timelock.git}"
# Known-good: python 0.0.2.dev0 + wasm wrapper 0.3.0 (356-byte W_time).
TIMELOCK_REV="${TIMELOCK_REV:-ccccca019409c89f31fd687352db8060bfb4aae6}"

timelock_stack_ok() {
  "$PY" -c '
from importlib.metadata import version
try:
    wasm = version("timelock_wasm_wrapper")
    py = version("timelock")
except Exception:
    raise SystemExit(1)
parts = [int(x) for x in wasm.split(".")[:2]]
raise SystemExit(0 if parts >= [0, 3] and py.startswith("0.0.2") else 1)
' >/dev/null 2>&1
}

# ── Choose interpreter ──────────────────────────────────────────────
# Preferred: python3.10. Fallbacks: python3.11, python3.12, python3.
# Timelock wasm is built from source for the active interpreter.
if [[ -n "${PY_BIN:-}" ]]; then
  command -v "$PY_BIN" >/dev/null 2>&1 || die "PY_BIN=$PY_BIN not found on PATH"
elif command -v python3.10 >/dev/null 2>&1; then
  PY_BIN="python3.10"
elif command -v python3.11 >/dev/null 2>&1; then
  PY_BIN="python3.11"
  warn "python3.10 not found, falling back to python3.11 -- timelock may need source build"
elif command -v python3.12 >/dev/null 2>&1; then
  PY_BIN="python3.12"
  warn "python3.10 not found, falling back to python3.12 -- timelock may need source build"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
  warn "python3.10 not found, falling back to generic python3"
else
  die "No Python 3 interpreter found. Install python3.10 or set PY_BIN."
fi

# Validate the chosen interpreter actually works
"$PY_BIN" -c "import sys; assert sys.version_info >= (3, 10), f'Need Python >= 3.10, got {sys.version}'" \
  || die "$PY_BIN is older than 3.10 -- please install python3.10+"

# Verify venv module is available
"$PY_BIN" -c "import venv" 2>/dev/null \
  || die "$PY_BIN lacks the venv module. Install python3-venv (e.g. apt install python3.10-venv)."

SUDO=""
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  SUDO="sudo"
fi

PY_VERSION="$("$PY_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")')"
step "Using interpreter: $PY_BIN (Python $PY_VERSION)"

# ── Create (or reuse) venv ──────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  step "Creating virtualenv at $VENV"
  "$PY_BIN" -m venv "$VENV"
  ok "Created $VENV"
else
  EXISTING_PY="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "unknown")"
  WANTED_PY="$("$PY_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  if [[ "$EXISTING_PY" != "$WANTED_PY" ]]; then
    warn "Existing venv is Python $EXISTING_PY but we want $WANTED_PY -- recreating"
    rm -rf "$VENV"
    "$PY_BIN" -m venv "$VENV"
    ok "Recreated $VENV with Python $WANTED_PY"
  else
    ok "Reusing existing $VENV (Python $EXISTING_PY)"
  fi
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PY="$VENV/bin/python"
PIP="$PY -m pip"

# Upgrade pip tooling -- pin setuptools for bittensor compatibility
step "Upgrading pip tooling"
$PY -m pip install -qU pip "setuptools~=70.0" wheel
ok "pip=$($PY -m pip --version | awk '{print $2}'), setuptools=$($PY -c 'import setuptools; print(setuptools.__version__)')"

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

# Live subnet W_time is 356 bytes. Skip PyPI (old 372-byte stack).
step "Checking timelock stack (need wasm wrapper >= 0.3.0)"
NEED_TIMELOCK=1
if timelock_stack_ok; then
  ok "timelock 0.0.2.dev0 + wasm wrapper >= 0.3.0 already installed"
  NEED_TIMELOCK=0
fi

if [[ "$NEED_TIMELOCK" -eq 1 ]]; then
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
  rustup toolchain install stable 2>/dev/null
  rustup default stable 2>/dev/null

  step "Cloning/updating timelock sources ($TIMELOCK_REV)"
  if [[ ! -d "$SRC/.git" ]]; then
    git clone "$TIMELOCK_GIT" "$SRC"
  else
    git -C "$SRC" remote set-url origin "$TIMELOCK_GIT"
  fi
  git -C "$SRC" fetch origin "$TIMELOCK_REV"
  git -C "$SRC" checkout --detach FETCH_HEAD

  # Backward-compat fix for ark_std rename (no-op if not needed)
  if [[ -d "$SRC/wasm/src" ]]; then
    sed -i.bak 's|ark_std::rand::rng::OsRng|ark_std::rand::rngs::OsRng|g' "$SRC/wasm/src/"{py,js}.rs || true
    # Fix Identity::new borrow: upstream changed signature to expect &[u8]
    sed -i 's|Identity::new(b"", id)|Identity::new(b"", \&id)|g' "$SRC/wasm/src/"{py,js}.rs || true
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

timelock_stack_ok || die "timelock stack must be 0.0.2.dev0 + wasm wrapper >= 0.3.0 (not PyPI 0.0.1.dev0)"

# ── Project requirements ────────────────────────────────────────────
if [[ -f "$ROOT/requirements.txt" ]]; then
  step "Installing project requirements into .venv"
  MAX_RETRIES=3
  for attempt in $(seq 1 $MAX_RETRIES); do
    if $PY -m pip install -r "$ROOT/requirements.txt"; then
      ok "Requirements installed (attempt $attempt/$MAX_RETRIES)"
      break
    fi
    if [[ $attempt -lt $MAX_RETRIES ]]; then
      warn "pip install failed (attempt $attempt/$MAX_RETRIES) -- retrying in 5s"
      sleep 5
    else
      die "Failed to install requirements after $MAX_RETRIES attempts"
    fi
  done
else
  warn "No requirements.txt found – skipping"
fi

# ── Validate key imports ────────────────────────────────────────────
step "Verifying critical packages import correctly"
FAILED_IMPORTS=()
for pkg in numpy scipy sklearn pandas torch bittensor requests aiohttp shap xgboost timelock; do
  if ! $PY -c "import $pkg" 2>/dev/null; then
    FAILED_IMPORTS+=("$pkg")
  fi
done

if [[ ${#FAILED_IMPORTS[@]} -gt 0 ]]; then
  die "The following packages failed to import: ${FAILED_IMPORTS[*]}"
fi
ok "All critical packages import successfully"

# ── BLAS library pin ────────────────────────────────────────────────
# numpy==2.2.6 wheels bundle scipy-openblas (OpenBLAS 0.3.29).
# We assert this at install time so a wheel swap or channel change is caught.
EXPECTED_BLAS_NAME="scipy-openblas"
EXPECTED_BLAS_VER="0.3.29"

step "Verifying pinned BLAS library ($EXPECTED_BLAS_NAME $EXPECTED_BLAS_VER)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  $PY -c "
import numpy as np, sys, os

print('  BLAS env vars:')
for v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MKL_CBWR',
          'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    print(f'    {v}={os.environ.get(v, \"(unset)\")}')

expected_name = '$EXPECTED_BLAS_NAME'
expected_ver  = '$EXPECTED_BLAS_VER'

cfg = np.show_config('dicts') if hasattr(np, 'show_config') else None
if not isinstance(cfg, dict):
    print('  WARNING: numpy.show_config(\"dicts\") unavailable, cannot verify BLAS pin')
    np.show_config()
    sys.exit(0)

blas_info  = cfg.get('Build Dependencies', {}).get('blas', {})
lapack_info = cfg.get('Build Dependencies', {}).get('lapack', {})
blas_name = blas_info.get('name', 'unknown')
blas_ver  = blas_info.get('version', 'unknown')
print(f'  numpy BLAS:   {blas_name} {blas_ver}')
print(f'  numpy LAPACK: {lapack_info.get(\"name\",\"unknown\")} {lapack_info.get(\"version\",\"unknown\")}')

if blas_name != expected_name:
    print(f'  FATAL: expected BLAS \"{expected_name}\" but got \"{blas_name}\"', file=sys.stderr)
    print(f'         Ensure numpy=={np.__version__} is installed from PyPI (not intel, conda, etc.)', file=sys.stderr)
    sys.exit(1)
if blas_ver != expected_ver:
    print(f'  WARNING: expected BLAS version {expected_ver} but got {blas_ver}')
    print(f'           Weights may differ from reference -- update EXPECTED_BLAS_VER if intentional')

openblas_cfg = blas_info.get('openblas configuration', '')
print(f'  OpenBLAS config: {openblas_cfg}')

a = np.random.RandomState(42).randn(200, 200)
ref = a @ a.T
for _ in range(5):
    assert np.array_equal(ref, a @ a.T), 'BLAS matmul not bitwise reproducible!'
print('  matmul reproducibility: PASS (5/5 identical)')
"
ok "BLAS pin verified: $EXPECTED_BLAS_NAME $EXPECTED_BLAS_VER"

# ── Summary ─────────────────────────────────────────────────────────
echo
ok "Installation complete in $VENV"
$PY -c "
import sys, numpy, scipy, sklearn, pandas, torch
from importlib.metadata import version
print(f'  Python     {sys.version.split()[0]}')
print(f'  numpy      {numpy.__version__}')
print(f'  scipy      {scipy.__version__}')
print(f'  sklearn    {sklearn.__version__}')
print(f'  pandas     {pandas.__version__}')
print(f'  torch      {torch.__version__}')
print(f'  timelock   {version(\"timelock\")}')
print(f'  tlock-wasm {version(\"timelock_wasm_wrapper\")}')
"
echo "Activate with:  source \"$VENV/bin/activate\""
