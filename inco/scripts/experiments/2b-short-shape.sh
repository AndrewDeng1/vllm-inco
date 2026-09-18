#!/usr/bin/env bash
# 2b. ISL 128 / OSL 128 -- short enough that saturation becomes visible.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$M" run "$INCO/modal/modal_baseline.py" \
  --label short-128-128-v2 --isl 128 --osl 128 \
  --concurrencies "1,2,4,8,16,32,64,128,192,256,320,384,448" \
  --max-num-seqs 448 "$@"
