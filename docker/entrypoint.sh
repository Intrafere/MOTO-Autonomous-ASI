#!/bin/sh
set -eu

export MOTO_GENERIC_MODE="${MOTO_GENERIC_MODE:-true}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export MOTO_DATA_ROOT="${MOTO_DATA_ROOT:-/app/backend/data}"

mkdir -p "$MOTO_DATA_ROOT"

if [ -n "${MOTO_LOG_ROOT:-}" ]; then
    mkdir -p "$MOTO_LOG_ROOT"
fi

# `backend/api/main.py` remains the source of truth for hosted env validation.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec python -m backend.api.main
