#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

resolve_bootstrap_python() {
    if command -v python3 >/dev/null 2>&1; then
        local candidate
        candidate="$(command -v python3)"
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi
    if command -v python >/dev/null 2>&1; then
        local candidate
        candidate="$(command -v python)"
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi
    return 1
}

create_repo_venv() {
    if [[ -z "${BOOTSTRAP_PYTHON:-}" ]]; then
        BOOTSTRAP_PYTHON="$(resolve_bootstrap_python || true)"
    fi
    if [[ -z "${BOOTSTRAP_PYTHON:-}" ]]; then
        echo "ERROR: Python 3.10+ is required to launch MOTO on Ubuntu 24.04."
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
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    create_repo_venv
elif ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="$(resolve_bootstrap_python || true)"
    if [[ -z "${BOOTSTRAP_PYTHON:-}" ]]; then
        echo "ERROR: Existing repo-local .venv uses Python older than 3.10, and no replacement Python 3.10+ was found."
        echo "Install Python 3 and python3-venv, then run this launcher again."
        echo "Example: sudo apt install python3 python3-venv"
        exit 1
    fi
    echo "Existing repo-local .venv uses Python older than 3.10. Recreating it ..."
    rm -rf "$VENV_DIR"
    create_repo_venv
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Expected launcher interpreter at $PYTHON_BIN"
    exit 1
fi

export MOTO_LAUNCHER_ENTRYPOINT="$SCRIPT_DIR/linux-ubuntu-launcher.sh"
exec "$PYTHON_BIN" "$SCRIPT_DIR/moto_launcher.py" "$@"
