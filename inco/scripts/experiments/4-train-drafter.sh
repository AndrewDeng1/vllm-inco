#!/usr/bin/env bash
# 4. Train the DFlash2 drafter against the REAP-50% target, on-policy. ~65 min.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$M" run --detach "$INCO/modal/modal_speculators.py" \
  --label dflash2-reap50-code-5k "$@"
