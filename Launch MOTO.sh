#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

resolve_bootstrap_python() {
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    return 1
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    BOOTSTRAP_PYTHON="$(resolve_bootstrap_python || true)"
    if [[ -z "${BOOTSTRAP_PYTHON:-}" ]]; then
        echo "ERROR: Python 3.8+ is required to launch MOTO on Ubuntu 24.04."
        echo "Install Python 3 and python3-venv, then run this launcher again."
        echo "Example: sudo apt install python3 python3-venv"
        exit 1
    fi

    echo "Creating repo-local Python environment in .venv ..."
    if ! "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"; then
        echo "ERROR: Failed to create the repo-local Python environment."
        echo "On Ubuntu 24.04, ensure python3-venv is installed:"
        echo "  sudo apt install python3-venv"
        exit 1
    fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Expected launcher interpreter at $PYTHON_BIN"
    exit 1
fi

export MOTO_LAUNCHER_ENTRYPOINT="$SCRIPT_DIR/Launch MOTO.sh"
exec "$PYTHON_BIN" "$SCRIPT_DIR/moto_launcher.py" "$@"
