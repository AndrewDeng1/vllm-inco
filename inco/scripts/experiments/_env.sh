# Shared paths for the experiment scripts. Sourced, never run directly.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INCO="$REPO/inco"
M="$REPO/.venv/bin/modal"
PY="$REPO/.venv/bin/python"

BASE="Qwen/Qwen3-30B-A3B-Instruct-2507"
# The inco-reap volume mounts at /reap in modal_baseline.py and
# modal_speculators.py, but at /artifacts in modal_reap.py. Same checkpoint,
# two paths -- use the one that matches the entrypoint you are calling.
_PRUNED_REL="pruned/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/pruned_models/layerwise_reap-renorm_true-seed_42-0.50"
PRUNED="/reap/$_PRUNED_REL"
PRUNED_ARTIFACTS="/artifacts/$_PRUNED_REL"

# Volume downloads are re-runnable.
get() { "$M" volume get --force "$@"; }
