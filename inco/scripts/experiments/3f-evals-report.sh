#!/usr/bin/env bash
# 3f. Collect the eval results once 3f-evals.sh has finished and diff them.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

get inco-reap evalplus "$INCO/results/evalplus"
get inco-reap evals "$INCO/results/evals"
cd "$INCO" && "$PY" -m bench.eval_compare \
  Qwen3-30B-A3B-Instruct-2507 layerwise_reap-renorm_true-seed_42-0.50 \
  --eval-root results/evals/evals
