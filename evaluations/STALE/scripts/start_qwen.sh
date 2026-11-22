#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VOLUME_ROOT="${LIGHTMEM_STALE_ROOT:-/workspace/lightmem-stale}"
PORT="${QWEN_PORT:-8000}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-16384}"

export UV_PROJECT_ENVIRONMENT="${VOLUME_ROOT}/.venv"
export UV_CACHE_DIR="${VOLUME_ROOT}/cache/uv"
export HF_HOME="${VOLUME_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

mkdir -p "${VOLUME_ROOT}/logs"
cd "${EVAL_ROOT}"

exec uv run vllm serve "${VOLUME_ROOT}/models/qwen3-4b" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.60 \
  --seed 42 \
  --reasoning-parser qwen3

