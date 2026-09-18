#!/usr/bin/env bash
# 4b. The same A/B on the unpruned target with its own drafter.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$M" run --detach "$INCO/modal/modal_speculators.py::acceptance" \
  --label specdec-dense-v5 --model "$BASE" \
  --drafter /spec/dflash2-dense-code-5k/checkpoints/4 \
  --kv-cache-gib 10 --gpu-memory-utilization 0.92 "$@"
