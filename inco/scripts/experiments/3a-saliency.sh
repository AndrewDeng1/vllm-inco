#!/usr/bin/env bash
# 3a. Saliency heatmaps from the observation dump. No GPU needed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

get inco-reap \
  pruned/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/layerwise/observations_1024_cosine-seed_42.pt \
  "$INCO/results/saliency/"
cd "$INCO" && "$PY" scripts/plot_reap_saliency.py
