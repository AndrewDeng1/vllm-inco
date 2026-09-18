#!/usr/bin/env bash
# 1. Baseline sweep, ISL 1024 / OSL 512. ~19 min.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$M" run "$INCO/modal/modal_baseline.py" \
  --label chat-1024-512 --isl 1024 --osl 512 \
  --concurrencies "1,2,4,8,16,24,32,40,48,56,64,72,80" \
  --max-num-seqs 80 "$@"
