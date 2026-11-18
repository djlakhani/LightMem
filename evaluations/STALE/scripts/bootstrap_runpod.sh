#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VOLUME_ROOT="${LIGHTMEM_STALE_ROOT:-/workspace/lightmem-stale}"

export UV_PROJECT_ENVIRONMENT="${VOLUME_ROOT}/.venv"
export UV_CACHE_DIR="${VOLUME_ROOT}/cache/uv"
export HF_HOME="${VOLUME_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Some RunPod images enable the optional hf_transfer backend globally. The
# evaluation environment intentionally does not depend on that package, so do
# not let an inherited image setting break Hugging Face downloads.
unset HF_HUB_ENABLE_HF_TRANSFER

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${VOLUME_ROOT}/runs"
mkdir -p "${VOLUME_ROOT}/models"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "${EVAL_ROOT}"
if [[ -f uv.lock ]]; then
  uv sync --locked --group dev
else
  uv lock
  uv sync --group dev
fi

VOLUME_ROOT="${VOLUME_ROOT}" uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download

root = os.environ["VOLUME_ROOT"]
models = (
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "qwen3-4b"),
    ("microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", "5f0c82792b7ea14c6484e015b6a072009496b7f2", "llmlingua-2"),
    ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "all-MiniLM-L6-v2"),
)
for model, revision, directory in models:
    destination = os.path.join(root, "models", directory)
    print(f"Caching {model}@{revision} at {destination}")
    snapshot_download(model, revision=revision, local_dir=destination)
PY

echo "RunPod environment is ready at ${UV_PROJECT_ENVIRONMENT}"

