#!/usr/bin/env bash
# 2. Dense 4B control for the MoE-vs-dense scaling comparison, then plot.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$M" run "$INCO/modal/modal_baseline.py" \
  --label dense-4b-unpinned-1024-512 --model Qwen/Qwen3-4B-Instruct-2507 \
  --isl 1024 --osl 512 --kv-cache-gib 0 --max-num-seqs 256 \
  --concurrencies "1,2,4,8,16,24,32,40,48,56,64,72,80,96,112,128,160,192,224,256" "$@"

get inco-results dense-4b-unpinned-1024-512 "$INCO/results"
cd "$INCO" && "$PY" scripts/plot_moe_vs_dense_ratio.py
