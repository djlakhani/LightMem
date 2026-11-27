#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/smoke_lightmem.sh [OPTIONS]

Run two STALE scenarios through the complete LightMem pipeline using a local
Qwen/vLLM server.

Options:
  --context-view VIEW   Input view: evidence_only (default) or truncated.
  --retrieval-limit N   Number of embedding results per question (default: 20).
  --show-attributes     Print the two stored attribute labels for each result.
  --force-selection-fallback
                        Corrupt selection responses to test embedding fallback.
  -h, --help            Show this help and exit.

Environment:
  LIGHTMEM_STALE_ROOT   Persistent cache/run root (default: /workspace/lightmem-stale).
  QWEN_PORT             Local vLLM port (default: 8000).
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

CONTEXT_VIEW="evidence_only"
RETRIEVAL_LIMIT=20
SHOW_ATTRIBUTES=0
FORCE_SELECTION_FALLBACK=0

while (( $# > 0 )); do
  case "$1" in
    --context-view)
      (( $# >= 2 )) || die "--context-view requires a value"
      CONTEXT_VIEW="$2"
      shift 2
      ;;
    --context-view=*)
      CONTEXT_VIEW="${1#*=}"
      shift
      ;;
    --show-attributes)
      SHOW_ATTRIBUTES=1
      shift
      ;;
    --retrieval-limit)
      (( $# >= 2 )) || die "--retrieval-limit requires a value"
      RETRIEVAL_LIMIT="$2"
      shift 2
      ;;
    --retrieval-limit=*)
      RETRIEVAL_LIMIT="${1#*=}"
      shift
      ;;
    --force-selection-fallback)
      FORCE_SELECTION_FALLBACK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help for usage)"
      ;;
  esac
done

case "${CONTEXT_VIEW}" in
  evidence_only|truncated) ;;
  *) die "--context-view must be evidence_only or truncated" ;;
esac

[[ "${RETRIEVAL_LIMIT}" =~ ^[0-9]+$ ]] || die "--retrieval-limit must be a positive integer"
(( RETRIEVAL_LIMIT >= 1 )) || die "--retrieval-limit must be a positive integer"

QWEN_PORT="${QWEN_PORT:-8000}"
[[ "${QWEN_PORT}" =~ ^[0-9]+$ ]] || die "QWEN_PORT must be an integer from 1 to 65535"
(( QWEN_PORT >= 1 && QWEN_PORT <= 65535 )) || die "QWEN_PORT must be an integer from 1 to 65535"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VOLUME_ROOT="${LIGHTMEM_STALE_ROOT:-/workspace/lightmem-stale}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${VOLUME_ROOT}/smoke-runs/${RUN_STAMP}"
RUN_LOG="${RUN_DIR}/smoke.log"
SERVER_LOG="${RUN_DIR}/qwen-vllm.log"

mkdir -p \
  "${VOLUME_ROOT}/bin" \
  "${VOLUME_ROOT}/cache/uv" \
  "${VOLUME_ROOT}/cache/huggingface" \
  "${VOLUME_ROOT}/models" \
  "${RUN_DIR}"

# Keep a timestamped transcript while still showing progress interactively.
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "STALE LightMem smoke run: ${RUN_STAMP}"
echo "Context view: ${CONTEXT_VIEW}"
echo "Retrieval limit: ${RETRIEVAL_LIMIT}"
echo "Force selection fallback: ${FORCE_SELECTION_FALLBACK}"
echo "Run directory: ${RUN_DIR}"

export PATH="${VOLUME_ROOT}/bin:${PATH}"
export UV_PROJECT_ENVIRONMENT="${VOLUME_ROOT}/.venv"
export UV_CACHE_DIR="${VOLUME_ROOT}/cache/uv"
export HF_HOME="${VOLUME_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM="false"
unset HF_HUB_ENABLE_HF_TRANSFER
unset OPENAI_API_KEY

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is missing; installing it under ${VOLUME_ROOT}/bin"
  curl -LsSf https://astral.sh/uv/install.sh -o "${RUN_DIR}/uv-installer.sh"
  env UV_INSTALL_DIR="${VOLUME_ROOT}/bin" sh "${RUN_DIR}/uv-installer.sh"
fi

cd "${EVAL_ROOT}"
echo "Syncing the locked STALE environment"
uv sync --locked --group dev

export VOLUME_ROOT
uv run --no-sync python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(os.environ["VOLUME_ROOT"])
models = (
    (
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c",
        "qwen3-4b",
    ),
    (
        "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        "5f0c82792b7ea14c6484e015b6a072009496b7f2",
        "llmlingua-2",
    ),
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "all-MiniLM-L6-v2",
    ),
)


def present(path: Path) -> bool:
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    return (path / "config.json").is_file() and bool(weights)


for model, revision, directory in models:
    destination = root / "models" / directory
    if present(destination):
        print(f"Using cached model: {destination}")
        continue
    print(f"Downloading {model}@{revision} to {destination}")
    snapshot_download(model, revision=revision, local_dir=destination)
    if not present(destination):
        raise RuntimeError(f"Model download is incomplete: {destination}")
PY

QWEN_MODEL="${VOLUME_ROOT}/models/qwen3-4b"
HEALTH_URL="http://127.0.0.1:${QWEN_PORT}/health"
QWEN_PID=""
STARTED_SERVER=0

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "${STARTED_SERVER}" == "1" && -n "${QWEN_PID}" ]]; then
    echo "Stopping owned vLLM server (PID ${QWEN_PID})"
    kill "${QWEN_PID}" 2>/dev/null || true
    wait "${QWEN_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

if curl --fail --silent --show-error --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
  echo "Reusing healthy vLLM server on port ${QWEN_PORT}"
else
  echo "Starting vLLM on port ${QWEN_PORT}; log: ${SERVER_LOG}"
  uv run --no-sync vllm serve "${QWEN_MODEL}" \
    --served-model-name "${QWEN_MODEL}" \
    --host 127.0.0.1 \
    --port "${QWEN_PORT}" \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.60 \
    --max-num-seqs 8 \
    --seed 42 \
    --reasoning-parser qwen3 \
    >"${SERVER_LOG}" 2>&1 &
  QWEN_PID=$!
  STARTED_SERVER=1

  SERVER_HEALTHY=0
  for ((attempt = 1; attempt <= 180; attempt++)); do
    if curl --fail --silent --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
      SERVER_HEALTHY=1
      break
    fi
    if ! kill -0 "${QWEN_PID}" 2>/dev/null; then
      echo "vLLM exited before becoming healthy. Last log lines:" >&2
      tail -n 80 "${SERVER_LOG}" >&2 || true
      exit 1
    fi
    sleep 2
  done
  if [[ "${SERVER_HEALTHY}" != "1" ]]; then
    echo "vLLM did not become healthy within 360 seconds. Last log lines:" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  echo "Owned vLLM server is healthy (PID ${QWEN_PID})"
fi

export CONTEXT_VIEW RETRIEVAL_LIMIT SHOW_ATTRIBUTES FORCE_SELECTION_FALLBACK
export RUN_DIR QWEN_MODEL QWEN_PORT EVAL_ROOT
uv run --no-sync python - <<'PY'
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from lightmem.memory.attributes import ATTRIBUTE_TO_GROUP, validate_attribute_pair
from stale_lightmem.backend import LightMemBackend, diff_memories
from stale_lightmem.config import EvalConfig, LightMemConfig, PathsConfig, QwenConfig, S3Config
from stale_lightmem.dataset import DIMENSIONS, load_dataset
from stale_lightmem.qwen import (
    NoThinkingQwenClient,
    answer_messages,
    assert_no_thinking,
    select_relevant_memories,
)

context_view = os.environ["CONTEXT_VIEW"]
retrieval_limit = int(os.environ["RETRIEVAL_LIMIT"])
show_attributes = os.environ["SHOW_ATTRIBUTES"] == "1"
force_selection_fallback = os.environ["FORCE_SELECTION_FALLBACK"] == "1"
run_dir = Path(os.environ["RUN_DIR"])
eval_root = Path(os.environ["EVAL_ROOT"])
qwen_model = os.environ["QWEN_MODEL"]
port = int(os.environ["QWEN_PORT"])
scenario_path = eval_root / "scenarios.json"

config = EvalConfig(
    schema_version=1,
    scenario_path=str(scenario_path),
    scenario_sha256="",
    context_view=context_view,
    paths=PathsConfig(
        volume_root=str(run_dir.parent.parent),
        run_root=str(run_dir.parent),
        cache_root=str(run_dir.parent.parent / "cache"),
    ),
    qwen=QwenConfig(
        model=qwen_model,
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="EMPTY",
        thinking=False,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        extraction_max_tokens=2048,
        answer_max_tokens=512,
    ),
    lightmem=LightMemConfig(
        compression_model=str(run_dir.parent.parent / "models" / "llmlingua-2"),
        compression_revision="5f0c82792b7ea14c6484e015b6a072009496b7f2",
        compression_rate=0.8,
        embedding_model=str(run_dir.parent.parent / "models" / "all-MiniLM-L6-v2"),
        embedding_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        embedding_dimensions=384,
        messages_use="user_only",
        extraction_mode="flat",
        update_queue_top_k=20,
        update_queue_keep_n=10,
        update_score_threshold=0.8,
        retrieval_limit=retrieval_limit,
    ),
    s3=S3Config(enabled=False),
    source_path=str(eval_root / "smoke-inline-config"),
)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def validate_memories(memories: list[dict[str, Any]], phase: str) -> dict[str, list[str]]:
    if not memories:
        raise RuntimeError(f"{phase}: no stored memories")
    validated: dict[str, list[str]] = {}
    for memory in memories:
        memory_id = memory["id"]
        attributes = memory.get("payload", {}).get("attributes")
        canonical = validate_attribute_pair(attributes)
        if canonical is None or attributes != canonical:
            raise RuntimeError(
                f"{phase}: memory {memory_id} has malformed attributes: {attributes!r}"
            )
        if ATTRIBUTE_TO_GROUP[canonical[0]] == ATTRIBUTE_TO_GROUP[canonical[1]]:
            raise RuntimeError(
                f"{phase}: memory {memory_id} attributes are from the same group"
            )
        validated[memory_id] = canonical
    return validated


def subtract_stats(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after.get(key, 0) - before.get(key, 0) for key in sorted(set(after) | set(before))}


dataset = load_dataset(scenario_path)
scenarios = dataset.scenarios[:2]
if len(scenarios) != 2:
    raise RuntimeError(f"Expected at least two scenarios, found {len(scenarios)}")

class ForcedInvalidSelectionQwen:
    """Smoke-only proxy that corrupts selection JSON after a real Qwen call."""

    def __init__(self, inner: Any):
        self.inner = inner

    def generate(self, messages: list[dict[str, str]], **kwargs: Any):
        generation = self.inner.generate(messages, **kwargs)
        is_selection = bool(
            messages
            and messages[0].get("role") == "system"
            and messages[0].get("content", "").startswith(
                "You are an evidence selector."
            )
        )
        if is_selection:
            return replace(generation, content='{"memory_ids": []}')
        return generation


base_qwen = NoThinkingQwenClient(config.qwen)
print("Checking the Qwen OpenAI-compatible endpoint and no-thinking behavior")
base_qwen.preflight()
qwen: Any = (
    ForcedInvalidSelectionQwen(base_qwen)
    if force_selection_fallback
    else base_qwen
)
if force_selection_fallback:
    print("Selection fault injection enabled: both LLM attempts will be invalid")

backend = LightMemBackend(config, run_dir)
completed: list[dict[str, Any]] = []
try:
    for scenario_index, scenario in enumerate(scenarios, start=1):
        scenario_started = time.perf_counter()
        backend.reset()
        attribute_before = backend.memory.get_attribute_statistics()

        ingest = backend.ingest(scenario, context_view)
        before = backend.snapshot()
        before_attributes = validate_memories(before, f"scenario {scenario_index} before consolidation")

        consolidation_started = time.perf_counter()
        consolidation = backend.consolidate()
        consolidation_seconds = time.perf_counter() - consolidation_started
        after = backend.snapshot()
        after_attributes = validate_memories(after, f"scenario {scenario_index} after consolidation")
        for memory_id in sorted(set(before_attributes) & set(after_attributes)):
            if after_attributes[memory_id] != before_attributes[memory_id]:
                raise RuntimeError(
                    f"scenario {scenario_index}: surviving memory {memory_id} changed attributes "
                    f"from {before_attributes[memory_id]!r} to {after_attributes[memory_id]!r}"
                )

        answers: dict[str, Any] = {}
        retrieval_total = 0.0
        evidence_selection_total = 0.0
        answer_generation_total = 0.0
        selection_fallback_count = 0
        evidence_selection_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        answer_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for dimension_key, dimension, query_key in DIMENSIONS:
            question = scenario["probing_queries"][query_key]
            retrieval_started = time.perf_counter()
            formatted_memories, retrieval = backend.retrieve(
                question, config.lightmem.retrieval_limit
            )
            retrieval_seconds = time.perf_counter() - retrieval_started
            retrieval_total += retrieval_seconds
            if not retrieval or not formatted_memories:
                raise RuntimeError(
                    f"scenario {scenario_index} {dimension}: retrieval returned no memories"
                )
            if len(retrieval) != len(formatted_memories):
                raise RuntimeError(
                    f"scenario {scenario_index} {dimension}: retrieval trace is inconsistent"
                )

            print()
            print(
                f"Scenario {scenario_index} | UID: {scenario['uid']} | "
                f"Type: {scenario['type']} ({scenario['type_name']}) | "
                f"Question dimension: {dimension} ({dimension_key})"
            )
            print("Question:")
            print(question)
            print(
                f"Retrieved memories ({len(retrieval)}; "
                f"top-{config.lightmem.retrieval_limit} requested):"
            )
            for rank, item in enumerate(retrieval, start=1):
                payload = item.get("payload", {})
                print(
                    f"  {rank}. ID={item.get('id')} | score={item.get('score')!r} | "
                    f"timestamp={payload.get('time_stamp', '')}"
                )
                if show_attributes:
                    print(f"     attributes={payload.get('attributes')!r}")
                print(f"     {payload.get('memory', '')}")

            selected, selection_messages, selection = select_relevant_memories(
                qwen, question, retrieval
            )
            evidence_selection_total += selection.elapsed_seconds
            for key in evidence_selection_usage:
                evidence_selection_usage[key] += selection.usage.get(key, 0)
            selection_fallback_count += int(selection.fallback_used)
            selected_ids = [str(item["id"]) for item in selected]
            print(
                f"Evidence selection (method={selection.method}, "
                f"llm_attempts={selection.llm_attempts}):"
            )
            if selection.fallback_used:
                print(f"  fallback_reason={selection.fallback_reason}")
            for rank, memory_id in enumerate(selected_ids, start=1):
                print(f"  {rank}. {memory_id}")

            generation = qwen.generate(
                answer_messages(question, selected, dimension),
                max_tokens=config.qwen.answer_max_tokens,
            )
            assert_no_thinking(generation.content)
            if not generation.content.strip():
                raise RuntimeError(
                    f"scenario {scenario_index} {dimension}: Qwen returned an empty answer"
                )
            answer_generation_total += generation.elapsed_seconds
            for key in answer_usage:
                answer_usage[key] += generation.usage.get(key, 0)

            print("Qwen response:")
            print(generation.content)
            print(
                f"Question timing: retrieval={retrieval_seconds:.3f}s, "
                f"evidence_selection={selection.elapsed_seconds:.3f}s, "
                f"generation={generation.elapsed_seconds:.3f}s"
            )
            answers[dimension_key] = {
                "dimension": dimension,
                "query": question,
                "retrieved_memories": retrieval,
                "retrieval_elapsed_seconds": retrieval_seconds,
                "evidence_selection": {
                    "memory_ids": selected_ids,
                    "selected_memories": selected,
                    "prompt": selection_messages,
                    "generation": asdict(selection),
                },
                "answer": generation.content,
                "generation": asdict(generation),
            }

        attribute_after = backend.memory.get_attribute_statistics()
        attribute_stats = subtract_stats(attribute_after, attribute_before)
        token_stats = backend.statistics()
        timings = {
            "ingestion_seconds": ingest["elapsed_seconds"],
            "consolidation_seconds": consolidation_seconds,
            "retrieval_seconds": retrieval_total,
            "evidence_selection_seconds": evidence_selection_total,
            "answer_generation_seconds": answer_generation_total,
            "scenario_seconds": time.perf_counter() - scenario_started,
        }
        record = {
            "scenario": {
                "index": scenario_index,
                "uid": scenario["uid"],
                "type": scenario["type"],
                "type_name": scenario["type_name"],
                "context_view": context_view,
            },
            "ingest": ingest,
            "pre_update_memories": before,
            "consolidation": consolidation,
            "post_update_memories": after,
            "memory_diff": diff_memories(before, after),
            "answers": answers,
            "timings": timings,
            "token_statistics": {
                "lightmem": token_stats,
                "evidence_selection": evidence_selection_usage,
                "answer_generation": answer_usage,
            },
            "evidence_selection_statistics": {
                "questions": len(answers),
                "embedding_rank_fallbacks": selection_fallback_count,
            },
            "attribute_statistics": attribute_stats,
        }
        write_json(run_dir / f"scenario-{scenario_index:02d}.json", record)
        completed.append(record)

        print()
        print(f"Scenario {scenario_index} summary")
        print(f"  timings={json.dumps(timings, sort_keys=True)}")
        print(
            "  token_statistics="
            + json.dumps(record["token_statistics"], sort_keys=True, default=str)
        )
        print(
            "  attribute_label_and_fallback_statistics="
            + json.dumps(attribute_stats, sort_keys=True)
        )
        print(
            "  evidence_selection_statistics="
            + json.dumps(record["evidence_selection_statistics"], sort_keys=True)
        )
        print(
            "  memory_diff="
            + json.dumps(record["memory_diff"], sort_keys=True)
        )
finally:
    backend.close()

question_count = sum(len(item["answers"]) for item in completed)
selection_fallback_count = sum(
    item["evidence_selection_statistics"]["embedding_rank_fallbacks"]
    for item in completed
)
if len(completed) != 2 or question_count != 6:
    raise RuntimeError(
        f"Smoke run completed {len(completed)} scenarios and {question_count} questions; "
        "expected 2 scenarios and 6 questions"
    )
if force_selection_fallback and selection_fallback_count != question_count:
    raise RuntimeError(
        f"Forced fallback covered {selection_fallback_count} of {question_count} questions"
    )

write_json(
    run_dir / "summary.json",
    {
        "status": "pass",
        "context_view": context_view,
        "retrieval_limit": retrieval_limit,
        "show_attributes": show_attributes,
        "force_selection_fallback": force_selection_fallback,
        "scenario_count": len(completed),
        "question_count": question_count,
        "evidence_selection_fallbacks": selection_fallback_count,
        "scenario_files": [f"scenario-{index:02d}.json" for index in range(1, 3)],
    },
)
print()
print(f"Smoke artifacts: {run_dir}")
print("PASS: 2 scenarios, 6 questions")
PY
