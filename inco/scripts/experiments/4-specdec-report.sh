#!/usr/bin/env bash
# 4. Plot both acceptance runs once 4a and 4b have finished.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

get inco-spec specdec-reap50-v5 "$INCO/results/spec"
get inco-spec specdec-dense-v5 "$INCO/results/spec"
cd "$INCO" && "$PY" -m bench.plot_specdec \
  results/spec/specdec-reap50-v5/results.json \
  results/spec/specdec-dense-v5/results.json \
  --labels "REAP-50%,unpruned"
