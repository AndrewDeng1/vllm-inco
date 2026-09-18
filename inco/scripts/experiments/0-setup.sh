#!/usr/bin/env bash
# One-time setup: venv, Modal auth, gitignored clones, weight prefetch.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

[[ -x "$PY" ]] || uv venv --python 3.12 "$REPO/.venv"
uv pip install --python "$PY" modal -r "$INCO/requirements.txt"

[[ -f "$HOME/.modal.toml" ]] || "$M" setup  # browser OAuth

# Each clone installs its own vLLM, so each gets its own virtualenv.
[[ -d "$INCO/reap" ]] || git clone https://github.com/CerebrasResearch/reap.git "$INCO/reap"
[[ -d "$INCO/speculators" ]] || git clone https://github.com/vllm-project/speculators.git "$INCO/speculators"

"$M" run "$INCO/modal/modal_baseline.py::prefetch"  # caches 61GB of weights
