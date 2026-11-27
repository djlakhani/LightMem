#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VOLUME_ROOT="${LIGHTMEM_STALE_ROOT:-/workspace/lightmem-stale}"
CONFIG="${EVAL_ROOT}/configs/qwen3_4b_full.yaml"
PORT="${QWEN_PORT:-8000}"
LIMIT=5
RUN_ID=""
SKIP_BOOTSTRAP=0
NO_UPLOAD=0

usage() {
  cat <<'EOF'
Usage: scripts/run_telemetry_pilot.sh [options]

Run a resumable truncated-context STALE pilot with detailed filtering telemetry.

Options:
  --limit N          Number of scenarios (default: 5).
  --run-id ID        Reuse or create this run ID (default: timestamped pilot ID).
  --config PATH      Evaluation config (must use context_view: truncated).
  --skip-bootstrap   Reuse the existing uv environment and model cache.
  --no-upload        Disable S3 preflight and uploads.
  -h, --help         Show this help without requiring a GPU or credentials.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while (( $# > 0 )); do
  case "$1" in
    --limit)
      (( $# >= 2 )) || die "--limit requires a value"
      LIMIT="$2"
      shift 2
      ;;
    --limit=*) LIMIT="${1#*=}"; shift ;;
    --run-id)
      (( $# >= 2 )) || die "--run-id requires a value"
      RUN_ID="$2"
      shift 2
      ;;
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --config)
      (( $# >= 2 )) || die "--config requires a value"
      CONFIG="$2"
      shift 2
      ;;
    --config=*) CONFIG="${1#*=}"; shift ;;
    --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
    --no-upload) NO_UPLOAD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "${LIMIT}" =~ ^[0-9]+$ ]] || die "--limit must be a positive integer"
(( LIMIT >= 1 )) || die "--limit must be a positive integer"
[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"
[[ -n "${OPENAI_API_KEY:-}" ]] || die "OPENAI_API_KEY is required"

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="pilot-$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-id contains unsafe characters"

export UV_PROJECT_ENVIRONMENT="${VOLUME_ROOT}/.venv"
export UV_CACHE_DIR="${VOLUME_ROOT}/cache/uv"
export HF_HOME="${VOLUME_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

if (( SKIP_BOOTSTRAP == 0 )); then
  "${SCRIPT_DIR}/bootstrap_runpod.sh"
fi
export PATH="${HOME}/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || die "uv is unavailable; rerun without --skip-bootstrap"

mkdir -p "${VOLUME_ROOT}/logs" "${VOLUME_ROOT}/runs/${RUN_ID}/logs"
SERVER_OWNED=0
QWEN_PID=""

cleanup() {
  if (( SERVER_OWNED == 1 )) && [[ -n "${QWEN_PID}" ]]; then
    kill "${QWEN_PID}" 2>/dev/null || true
    wait "${QWEN_PID}" 2>/dev/null || true
  fi
  if (( NO_UPLOAD == 0 )) \
    && command -v uv >/dev/null 2>&1 \
    && [[ -f "${VOLUME_ROOT}/runs/${RUN_ID}/manifest.json" ]]; then
    (
      cd "${EVAL_ROOT}"
      uv run stale-lightmem --config "${CONFIG}" upload \
        --run-dir "${VOLUME_ROOT}/runs/${RUN_ID}"
    ) >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  "${SCRIPT_DIR}/start_qwen.sh" >"${VOLUME_ROOT}/runs/${RUN_ID}/logs/qwen-vllm.log" 2>&1 &
  QWEN_PID=$!
  SERVER_OWNED=1
  for _ in $(seq 1 240); do
    if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${QWEN_PID}" 2>/dev/null; then
      die "vLLM exited; inspect ${VOLUME_ROOT}/runs/${RUN_ID}/logs/qwen-vllm.log"
    fi
    sleep 2
  done
fi
curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null \
  || die "Qwen server did not become healthy on port ${PORT}"

cd "${EVAL_ROOT}"
ARGS=(
  uv run stale-lightmem
  --config "${CONFIG}"
  pilot
  --limit "${LIMIT}"
  --run-id "${RUN_ID}"
)
if (( NO_UPLOAD == 1 )); then
  ARGS+=(--no-upload)
fi

"${ARGS[@]}" 2>&1 | tee "${VOLUME_ROOT}/runs/${RUN_ID}/logs/pilot.log"

echo "Pilot run: ${VOLUME_ROOT}/runs/${RUN_ID}"
echo "Report: ${VOLUME_ROOT}/runs/${RUN_ID}/report.md"
echo "Detailed alias: ${VOLUME_ROOT}/runs/${RUN_ID}/memory_filtering_telemetry_report.md"
if (( NO_UPLOAD == 0 )); then
  echo "S3: s3://stale-qwen-evals-589498925064-us-east-1-an/lightmem/stale/qwen3-4b/${RUN_ID}/"
fi
