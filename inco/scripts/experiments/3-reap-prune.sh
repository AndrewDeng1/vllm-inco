#!/usr/bin/env bash
# 3. REAP 50% expert pruning, calibrated on codealpaca. ~90 min cold, ~3 min
# once observations are cached for the same sample count and seed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# batch_size is not a free speed knob: the observer materialises activations
# for all 128 experts, so 32 x 2048 is 68.7 GB and OOMs an 80GB card.
"$M" run "$INCO/modal/modal_reap.py" --compression-ratio 0.5 \
  --batches-per-category 128 --batch-size 8 --model-max-length 2048 "$@"
