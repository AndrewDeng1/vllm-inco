#!/usr/bin/env bash
# Harness test suite. No GPU required.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

cd "$INCO" && exec "$PY" -m pytest tests -q --cov=bench --cov-report=term-missing "$@"
