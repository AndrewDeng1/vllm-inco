#!/usr/bin/env bash
# 4b. The same drafter trained against the unpruned target, as a control.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$M" run --detach "$INCO/modal/modal_speculators.py" \
  --label dflash2-dense-code-5k --model "$BASE" "$@"
