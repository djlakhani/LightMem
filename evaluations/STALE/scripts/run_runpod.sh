#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VOLUME_ROOT="${LIGHTMEM_STALE_ROOT:-/workspace/lightmem-stale}"
CONFIG="${1:-${EVAL_ROOT}/configs/qwen3_4b_full.yaml}"
PORT="${QWEN_PORT:-8000}"

export UV_PROJECT_ENVIRONMENT="${VOLUME_ROOT}/.venv"
export UV_CACHE_DIR="${VOLUME_ROOT}/cache/uv"
export HF_HOME="${VOLUME_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

mkdir -p "${VOLUME_ROOT}/logs"
cd "${EVAL_ROOT}"

STARTED_SERVER=0
if ! curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  "${SCRIPT_DIR}/start_qwen.sh" >"${VOLUME_ROOT}/logs/qwen-vllm.log" 2>&1 &
  QWEN_PID=$!
  STARTED_SERVER=1
  for _ in $(seq 1 180); do
    if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${QWEN_PID}" 2>/dev/null; then
      echo "vLLM exited before becoming healthy; inspect ${VOLUME_ROOT}/logs/qwen-vllm.log" >&2
      exit 1
    fi
    sleep 2
  done
fi

cleanup() {
  if [[ "${STARTED_SERVER}" == "1" ]]; then
    kill "${QWEN_PID}" 2>/dev/null || true
    wait "${QWEN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

uv run stale-lightmem --config "${CONFIG}" all

