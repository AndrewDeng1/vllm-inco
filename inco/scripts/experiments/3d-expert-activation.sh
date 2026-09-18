#!/usr/bin/env bash
# 3d. Distinct experts touched per MoE layer vs batch size, both models.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$M" run "$INCO/modal/modal_reap.py::expert_activation" --model "$BASE"
"$M" run "$INCO/modal/modal_reap.py::expert_activation" --model "$PRUNED"

get inco-reap experts "$INCO/results/experts"
cd "$INCO" && "$PY" scripts/plot_expert_activation.py
