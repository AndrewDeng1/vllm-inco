#!/usr/bin/env bash
# 4a. Drafted vs undrafted A/B on the pruned target.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$M" run --detach "$INCO/modal/modal_speculators.py::acceptance" \
  --label specdec-reap50-v5 "$@"
