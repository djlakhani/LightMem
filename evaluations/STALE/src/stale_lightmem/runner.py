from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from .artifacts import (
    atomic_write_json,
    checkpoint_name,
    completed_checkpoint,
    read_json,
    write_jsonl,
)
from .config import EvalConfig
from .dataset import DIMENSIONS, ScenarioDataset, load_dataset
from .qwen import NoThinkingQwenClient, answer_messages
from .report import build_summary, write_report
from .s3 import S3ArtifactStore


def canonical_hash(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_sha(config: EvalConfig) -> str:
    repo = Path(config.source_path).resolve().parents[3]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in (
        "lightmem",
        "openai",
        "qdrant-client",
        "transformers",
        "torch",
        "vllm",
    ):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _resolved_hf_revision(model: str, requested_revision: str | None) -> str | None:
    try:
        from huggingface_hub import model_info

        return model_info(model, revision=requested_revision).sha
    except Exception:
        return requested_revision


def create_run_dir(config: EvalConfig, run_id: str | None = None) -> Path:
    git_sha = _git_sha(config)
    run_id = run_id or (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + f"-{git_sha[:8]}-{config.digest[:8]}"
    )
    run_dir = Path(config.paths.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_manifest(
    config: EvalConfig, dataset: ScenarioDataset, run_dir: Path
) -> dict[str, Any]:
    from .judge import SYSTEM_PROMPT_ALL_IN_ONE_JUDGE

    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "config_hash": config.digest,
        "config": config.to_dict(),
        "scenario_sha256": dataset.sha256,
        "scenario_count": len(dataset.scenarios),
        "prompt_version": "stale-author-prompts-v1",
        "judge_prompt_sha256": canonical_hash(SYSTEM_PROMPT_ALL_IN_ONE_JUDGE),
        "git_sha": _git_sha(config),
        "models": {
            "qwen": {
                "id": config.qwen.model,
                "revision": _resolved_hf_revision(
                    config.qwen.model, config.qwen.revision
                ),
                "thinking": False,
            },
            "compressor": {
                "id": config.lightmem.compression_model,
                "revision": _resolved_hf_revision(
                    config.lightmem.compression_model,
                    config.lightmem.compression_revision,
                ),
            },
            "embedder": {
                "id": config.lightmem.embedding_model,
                "revision": _resolved_hf_revision(
                    config.lightmem.embedding_model,
                    config.lightmem.embedding_revision,
                ),
            },
            "judge": {
                "id": config.judge.model,
                "reasoning_enabled": config.judge.reasoning_enabled,
            },
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }
    if config.dependency_resolution.enabled:
        from lightmem.memory.taxonomy import taxonomy_digest

        manifest["dependency_resolution"] = {
            "enabled": True,
            "taxonomy_sha256": taxonomy_digest(),
            "anchor_count": config.dependency_resolution.anchor_count,
            "semantic_candidates_per_anchor": (
                config.dependency_resolution.semantic_candidates_per_anchor
            ),
            "recent_candidates_per_anchor": (
                config.dependency_resolution.recent_candidates_per_anchor
            ),
            "max_dependencies_per_anchor": (
                config.dependency_resolution.max_dependencies_per_anchor
            ),
        }
    return manifest


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Run manifest not found: {path}")
    return read_json(path)


def _store(config: EvalConfig, run_dir: Path, upload: bool) -> S3ArtifactStore | None:
    if upload and config.s3.enabled:
        return S3ArtifactStore(config.s3, run_dir.name)
    return None


def preflight(
    config: EvalConfig, *, check_s3: bool = True, check_gpu: bool = True
) -> dict[str, Any]:
    dataset = load_dataset(config.resolved_scenario_path, config.scenario_sha256)
    Path(config.paths.run_root).mkdir(parents=True, exist_ok=True)
    if not os.access(config.paths.run_root, os.W_OK):
        raise RuntimeError(f"Run root is not writable: {config.paths.run_root}")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    gpu: dict[str, Any] = {"checked": check_gpu}
    if check_gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        properties = torch.cuda.get_device_properties(0)
        gpu.update(
            {
                "name": properties.name,
                "total_memory_gib": properties.total_memory / (1024**3),
            }
        )
        if properties.total_memory < 40 * 1024**3:
            raise RuntimeError(
                "Expected a GPU with at least 40 GiB for the full profile"
            )

    qwen = NoThinkingQwenClient(config.qwen)
    qwen_server = qwen.preflight()
    dependency_smoke: dict[str, Any] | None = None
    if config.dependency_resolution.enabled:
        from .dependency_resolution import smoke_dependency_resolution

        dependency_smoke = smoke_dependency_resolution(
            qwen, config.dependency_resolution
        )
    if check_s3 and config.s3.enabled:
        S3ArtifactStore(config.s3, "preflight-read-only").preflight()
    return {
        "scenario_count": len(dataset.scenarios),
        "scenario_sha256": dataset.sha256,
        "gpu": gpu,
        "qwen_no_thinking": True,
        "qwen_server": qwen_server,
        "dependency_resolution": dependency_smoke,
        "s3_checked": bool(check_s3 and config.s3.enabled),
    }


def run_generation(
    config: EvalConfig,
    run_dir: Path,
    *,
    limit: int | None = None,
    upload: bool = True,
    backend_factory: Any | None = None,
    qwen_client: Any | None = None,
) -> list[dict[str, Any]]:
    from .backend import diff_memories

    dataset = load_dataset(config.resolved_scenario_path, config.scenario_sha256)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("config_hash") != config.digest:
            raise ValueError("Run config hash does not match the existing manifest")
    else:
        manifest = make_manifest(config, dataset, run_dir)
        atomic_write_json(manifest_path, manifest)

    scenarios = dataset.scenarios[:limit] if limit else dataset.scenarios
    checkpoint_dir = run_dir / "checkpoints" / "generation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    store = _store(config, run_dir, upload)
    if store:
        store.upload_file(manifest_path, "manifest.json")
    qwen = qwen_client or NoThinkingQwenClient(config.qwen)
    if backend_factory is None:
        from .backend import LightMemBackend

        backend_factory = LightMemBackend
    backend = backend_factory(config, run_dir)
    try:
        for index, scenario in enumerate(scenarios, start=1):
            scenario_hash = canonical_hash(scenario)
            checkpoint = checkpoint_dir / checkpoint_name(index, scenario["uid"])
            if completed_checkpoint(checkpoint, config.digest, scenario_hash):
                continue
            started = time.perf_counter()
            try:
                backend.reset()
                ingest = backend.ingest(scenario, config.context_view)
                before = backend.snapshot()
                consolidation = backend.consolidate()
                after = backend.snapshot()
                answers: dict[str, Any] = {}
                for dim_key, dim_name, query_key in DIMENSIONS:
                    query = scenario["probing_queries"][query_key]
                    retrieval_started = time.perf_counter()
                    memories, trace = backend.retrieve(
                        query, config.lightmem.retrieval_limit
                    )
                    retrieval_seconds = time.perf_counter() - retrieval_started
                    answer_record: dict[str, Any] = {
                        "dimension": dim_name,
                        "query": query,
                        "retrieval": trace,
                        "retrieved_memories": memories,
                        "retrieval_elapsed_seconds": retrieval_seconds,
                    }
                    if config.dependency_resolution.enabled:
                        from .dependency_resolution import (
                            build_anchor_selection_messages,
                            build_final_answer_messages,
                            anchor_selection_response_format,
                            generate_dependency_resolution,
                            generate_validated_json,
                            parse_anchor_selection,
                        )

                        dependency = config.dependency_resolution
                        allowed_ids = {str(item["id"]) for item in trace}
                        anchor_messages = build_anchor_selection_messages(
                            query, trace, dim_name, dependency.anchor_count
                        )
                        anchor_started = time.perf_counter()
                        anchor_generation = generate_validated_json(
                            qwen,
                            anchor_messages,
                            lambda content: parse_anchor_selection(
                                content, allowed_ids, dependency.anchor_count
                            ),
                            max_tokens=dependency.anchor_max_tokens,
                            response_format=anchor_selection_response_format(
                                dependency.anchor_count, allowed_ids
                            ),
                        )
                        anchor_seconds = time.perf_counter() - anchor_started
                        trace_map = {str(item["id"]): item for item in trace}
                        anchor_ids = anchor_generation["parsed"]["anchor_ids"]
                        anchors = [trace_map[item] for item in anchor_ids]

                        candidate_started = time.perf_counter()
                        candidate_groups = {
                            str(anchor["id"]): backend.future_candidates(
                                anchor,
                                dependency.semantic_candidates_per_anchor,
                                dependency.recent_candidates_per_anchor,
                            )
                            for anchor in anchors
                        }
                        candidate_seconds = time.perf_counter() - candidate_started
                        resolution_started = time.perf_counter()
                        resolution_generation = generate_dependency_resolution(
                            qwen,
                            query,
                            anchors,
                            candidate_groups,
                            dependency.max_dependencies_per_anchor,
                            max_tokens=dependency.resolution_max_tokens,
                        )
                        resolution_seconds = time.perf_counter() - resolution_started
                        final_messages = build_final_answer_messages(
                            query,
                            dim_name,
                            anchors,
                            candidate_groups,
                            resolution_generation["parsed"],
                        )
                        generation = qwen.generate(
                            final_messages,
                            max_tokens=config.qwen.answer_max_tokens,
                        )
                        answer_record.update(
                            {
                                "anchor_selection": {
                                    "prompt": anchor_messages,
                                    "anchor_ids": anchor_ids,
                                    "anchors": anchors,
                                    "attempts": anchor_generation["attempts"],
                                    "elapsed_seconds": anchor_seconds,
                                },
                                "future_candidates_by_anchor": candidate_groups,
                                "future_candidate_retrieval_elapsed_seconds": (
                                    candidate_seconds
                                ),
                                "dependency_resolution": {
                                    "prompt": resolution_generation["prompt"],
                                    "mode": resolution_generation["mode"],
                                    "fallback_prompts": resolution_generation[
                                        "fallback_prompts"
                                    ],
                                    "result": resolution_generation["parsed"],
                                    "attempts": resolution_generation["attempts"],
                                    "batch_error": resolution_generation.get(
                                        "batch_error"
                                    ),
                                    "elapsed_seconds": resolution_seconds,
                                },
                                "prompt": final_messages,
                                "answer": generation.content,
                                "generation": asdict(generation),
                            }
                        )
                    else:
                        messages = answer_messages(query, memories, dim_name)
                        generation = qwen.generate(
                            messages,
                            max_tokens=config.qwen.answer_max_tokens,
                        )
                        answer_record.update(
                            {
                                "prompt": messages,
                                "answer": generation.content,
                                "generation": asdict(generation),
                            }
                        )
                    answers[dim_key] = answer_record
                record = {
                    "status": "complete",
                    "config_hash": config.digest,
                    "scenario_hash": scenario_hash,
                    "index": index,
                    "uid": scenario["uid"],
                    "type": scenario["type"],
                    "type_name": scenario["type_name"],
                    "context_view": config.context_view,
                    "ingest": ingest,
                    "pre_update_memories": before,
                    "consolidation": consolidation,
                    "post_update_memories": after,
                    "memory_diff": diff_memories(before, after),
                    "answers": answers,
                    "token_statistics": backend.statistics(),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                atomic_write_json(checkpoint, record)
                if store:
                    store.upload_file(checkpoint, str(checkpoint.relative_to(run_dir)))
            except Exception as exc:
                error = {
                    "status": "error",
                    "config_hash": config.digest,
                    "scenario_hash": scenario_hash,
                    "index": index,
                    "uid": scenario["uid"],
                    "type": scenario["type"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                atomic_write_json(checkpoint, error)
                if store:
                    store.upload_file(checkpoint, str(checkpoint.relative_to(run_dir)))
                raise
    finally:
        backend.close()

    records = [read_json(path) for path in sorted(checkpoint_dir.glob("*.json"))]
    completed = [record for record in records if record.get("status") == "complete"]
    write_jsonl(run_dir / "scenario_records.jsonl", completed)
    return completed


def run_judging(
    config: EvalConfig,
    run_dir: Path,
    *,
    upload: bool = True,
) -> list[dict[str, Any]]:
    from .judge import StaleJudge

    manifest = load_run_manifest(run_dir)
    if manifest.get("config_hash") != config.digest:
        raise ValueError("Run config hash does not match the judge configuration")
    dataset = load_dataset(config.resolved_scenario_path, config.scenario_sha256)
    scenario_map = {scenario["uid"]: scenario for scenario in dataset.scenarios}
    generation_dir = run_dir / "checkpoints" / "generation"
    generation = [read_json(path) for path in sorted(generation_dir.glob("*.json"))]
    generation = [record for record in generation if record.get("status") == "complete"]
    if not generation:
        raise RuntimeError("No completed generation checkpoints were found")

    judge = StaleJudge(config.judge)
    judge_dir = run_dir / "checkpoints" / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    store = _store(config, run_dir, upload)
    for record in generation:
        index = record["index"]
        uid = record["uid"]
        checkpoint = judge_dir / checkpoint_name(index, uid)
        if completed_checkpoint(checkpoint, config.digest, record["scenario_hash"]):
            continue
        try:
            judged = judge.evaluate(scenario_map[uid], record["answers"])
            result = {
                "status": "complete",
                "config_hash": config.digest,
                "scenario_hash": record["scenario_hash"],
                "index": index,
                "uid": uid,
                "type": record["type"],
                **judged,
            }
            atomic_write_json(checkpoint, result)
            if store:
                store.upload_file(checkpoint, str(checkpoint.relative_to(run_dir)))
        except Exception as exc:
            error = {
                "status": "error",
                "config_hash": config.digest,
                "scenario_hash": record["scenario_hash"],
                "index": index,
                "uid": uid,
                "type": record["type"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            atomic_write_json(checkpoint, error)
            if store:
                store.upload_file(checkpoint, str(checkpoint.relative_to(run_dir)))
            raise

    judgments = [read_json(path) for path in sorted(judge_dir.glob("*.json"))]
    completed = [record for record in judgments if record.get("status") == "complete"]
    write_jsonl(run_dir / "judgments.jsonl", completed)
    return completed


def generate_report(
    config: EvalConfig, run_dir: Path, *, allow_partial: bool = False
) -> dict[str, Any]:
    generation_dir = run_dir / "checkpoints" / "generation"
    judge_dir = run_dir / "checkpoints" / "judge"
    records = [read_json(path) for path in sorted(generation_dir.glob("*.json"))]
    records = [record for record in records if record.get("status") == "complete"]
    judgments = [read_json(path) for path in sorted(judge_dir.glob("*.json"))]
    judgments = [record for record in judgments if record.get("status") == "complete"]
    summary = build_summary(judgments, records)
    write_report(run_dir, summary)

    manifest = load_run_manifest(run_dir)
    manifest["status"] = summary["status"]
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(run_dir / "manifest.json", manifest)
    if summary["status"] != "complete" and not allow_partial:
        raise RuntimeError(
            f"Run is incomplete: judged {summary['judged_scenarios']} of 50 scenarios"
        )
    return summary


def upload_run(config: EvalConfig, run_dir: Path) -> list[str]:
    if not config.s3.enabled:
        return []
    return S3ArtifactStore(config.s3, run_dir.name).upload_run(run_dir)
