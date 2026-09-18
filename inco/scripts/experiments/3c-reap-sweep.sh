#!/usr/bin/env bash
# 3c. Sweep the pruned model with the freed memory re-pinned as KV cache,
# then overlay it on the baseline.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$M" run "$INCO/modal/modal_baseline.py" \
  --label reap50-1024-512 --served-model-name reap50 --model "$PRUNED" \
  --isl 1024 --osl 512 --kv-cache-gib 38 --max-num-seqs 256 \
  --concurrencies "1,2,4,8,16,24,32,40,48,56,64,72,80,96,128,160,192,224,256" "$@"

get inco-results reap50-1024-512 "$INCO/results"
cd "$INCO" && "$PY" -m bench.compare chat-1024-512 reap50-1024-512
