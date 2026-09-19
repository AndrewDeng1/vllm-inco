#!/usr/bin/env bash
# 3f. Pruning degradation: EvalPlus for code, lm-eval for multiple choice.
# Detached, so this returns once everything is queued.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

for model in "$BASE" "$PRUNED_ARTIFACTS"; do
  for dataset in humaneval mbpp; do  # one per call; two would OOM one container
    "$M" run --detach "$INCO/modal/modal_reap.py::evalplus_eval" \
      --model "$model" --dataset "$dataset" --greedy
  done
  "$M" run --detach "$INCO/modal/modal_reap.py::evaluate" \
    --model "$model" --tasks rte,openbookqa,winogrande,arc_challenge
done
