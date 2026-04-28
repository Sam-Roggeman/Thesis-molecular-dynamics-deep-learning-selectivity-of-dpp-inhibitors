#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-$ROOT_DIR/.venv_interpretability}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_TARGET="${TORCH_TARGET:-cuda}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found. Set PYTHON_BIN to a valid Python executable." >&2
  exit 1
fi

if [[ "$TORCH_TARGET" != "cuda" && "$TORCH_TARGET" != "cpu" ]]; then
  echo "Error: TORCH_TARGET must be 'cuda' or 'cpu'." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip

if [[ "$TORCH_TARGET" == "cuda" ]]; then
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
else
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
fi

python -m pip install -r "$ROOT_DIR/requirements-interpretability.txt"

echo "Done. Activate with: source $ENV_DIR/bin/activate"