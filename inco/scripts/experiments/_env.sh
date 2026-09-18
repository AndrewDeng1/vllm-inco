# Shared paths for the experiment scripts. Sourced, never run directly.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INCO="$REPO/inco"
M="$REPO/.venv/bin/modal"
PY="$REPO/.venv/bin/python"

BASE="Qwen/Qwen3-30B-A3B-Instruct-2507"
PRUNED="/artifacts/pruned/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/pruned_models/layerwise_reap-renorm_true-seed_42-0.50"

# Volume downloads are re-runnable.
get() { "$M" volume get --force "$@"; }
