from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import (
    atomic_write_json,
    checkpoint_name,
    read_json,
    read_jsonl,
    write_jsonl,
)
from .attribution import (
    DIMENSIONS,
    EXPERIMENT_VERSION,
    build_citation_prompt,
    citation_metrics,
    discover_state_candidates,
    parse_citation_response,
)
from .attribution_judge import AttributionJudge
from .config import EvalConfig, S3Config
from .dataset import load_dataset
from .qwen import NoThinkingQwenClient
from .s3 import S3ArtifactStore


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def evidence_s3_config(config: EvalConfig) -> S3Config:
    return replace(
        config.s3,
        prefix=f"{config.s3.prefix.strip('/')}/evidence-attribution",
    )


def experiment_hash(config: EvalConfig, source_manifest: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "version": EXPERIMENT_VERSION,
            "base_config_hash": config.digest,
            "source_run_id": source_manifest.get("run_id"),
            "source_config_hash": source_manifest.get("config_hash"),
            "source_scenario_hash": source_manifest.get("scenario_sha256"),
            "citation_count": 4,
            "citation_answer_max_tokens": max(config.qwen.answer_max_tokens, 768),
            "labeler_model": config.judge.model,
            "answer_model": config.qwen.model,
        }
    )


def attribution_run_root(config: EvalConfig) -> Path:
    return Path(config.paths.volume_root) / "evidence-attribution-runs"


def source_run_root(config: EvalConfig) -> Path:
    return Path(config.paths.volume_root) / "source-runs"


def create_attribution_run_dir(
    config: EvalConfig,
    source_manifest: dict[str, Any],
    run_id: str | None = None,
) -> Path:
    digest = experiment_hash(config, source_manifest)
    identifier = run_id or (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + f"-{str(source_manifest.get('run_id', 'source'))[:12]}-{digest[:8]}"
    )
    path = attribution_run_root(config) / identifier
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_source_run(
    config: EvalConfig,
    *,
    source_run_dir: str | Path | None = None,
    source_run_id: str | None = None,
) -> Path:
    if source_run_dir is not None:
        path = Path(source_run_dir).expanduser().resolve()
        if not (path / "scenario_records.jsonl").is_file():
            raise FileNotFoundError(f"Source scenario_records.jsonl not found: {path}")
        if not (path / "manifest.json").is_file():
            raise FileNotFoundError(f"Source manifest.json not found: {path}")
        return path
    if not source_run_id:
        raise ValueError("Provide either --source-run-dir or --source-run-id")
    destination = source_run_root(config) / source_run_id
    if (destination / "scenario_records.jsonl").is_file() and (
        destination / "manifest.json"
    ).is_file():
        return destination
    if not config.s3.enabled:
        raise RuntimeError("Cannot hydrate a source run because S3 is disabled")
    S3ArtifactStore(config.s3, source_run_id).download_run(destination)
    if not (destination / "scenario_records.jsonl").is_file():
        raise RuntimeError(f"Hydrated source run is incomplete: {destination}")
    return destination


def validate_source_run(
    config: EvalConfig, source_run_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = read_json(source_run_dir / "manifest.json")
    dataset = load_dataset(config.resolved_scenario_path, config.scenario_sha256)
    if manifest.get("scenario_sha256") != dataset.sha256:
        raise ValueError("Source run and local scenarios.json hashes do not match")
    records = read_jsonl(source_run_dir / "scenario_records.jsonl")
    if not records:
        raise RuntimeError("Source run has no scenario records")
    seen: set[str] = set()
    for record in records:
        uid = record.get("uid")
        if record.get("status") != "complete" or uid in seen:
            raise ValueError(f"Invalid or duplicate source record: {uid}")
        seen.add(uid)
        for dim_key, _, _ in DIMENSIONS:
            answer = record.get("answers", {}).get(dim_key, {})
            retrieval = answer.get("retrieval", [])
            if len(retrieval) < 4:
                raise ValueError(
                    f"Source {uid}/{dim_key} has fewer than four retrievals"
                )
            ids = [str(item.get("id")) for item in retrieval]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Source {uid}/{dim_key} has duplicate retrieval IDs")
    scenario_map = {item["uid"]: item for item in dataset.scenarios}
    missing = seen - set(scenario_map)
    if missing:
        raise ValueError(f"Source run has unknown scenario UIDs: {sorted(missing)}")
    return manifest, sorted(records, key=lambda item: item["index"]), scenario_map


def preflight_attribution(
    config: EvalConfig, *, check_s3: bool = True
) -> dict[str, Any]:
    load_dataset(config.resolved_scenario_path, config.scenario_sha256)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    NoThinkingQwenClient(config.qwen).preflight()
    if check_s3 and config.s3.enabled:
        S3ArtifactStore(config.s3, "preflight-read-only").preflight()
    return {
        "qwen_no_thinking": True,
        "judge_model": config.judge.model,
        "s3_checked": bool(check_s3 and config.s3.enabled),
        "experiment_version": EXPERIMENT_VERSION,
    }


def _store(config: EvalConfig, run_dir: Path, upload: bool) -> S3ArtifactStore | None:
    if upload and config.s3.enabled:
        return S3ArtifactStore(evidence_s3_config(config), run_dir.name)
    return None


def _make_manifest(
    config: EvalConfig,
    source_run_dir: Path,
    source_manifest: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    digest = experiment_hash(config, source_manifest)
    return {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_hash": digest,
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "source_run_id": source_manifest.get("run_id"),
        "source_run_dir": str(source_run_dir),
        "source_config_hash": source_manifest.get("config_hash"),
        "scenario_sha256": source_manifest.get("scenario_sha256"),
        "answer_model": config.qwen.model,
        "answer_thinking": False,
        "judge_model": config.judge.model,
        "judge_reasoning_enabled": config.judge.reasoning_enabled,
        "citation_count": 4,
        "citation_answer_max_tokens": max(config.qwen.answer_max_tokens, 768),
        "s3_prefix": evidence_s3_config(config).prefix,
    }


def ensure_manifest(
    config: EvalConfig, source_run_dir: Path, run_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest = read_json(source_run_dir / "manifest.json")
    expected = experiment_hash(config, source_manifest)
    path = run_dir / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest.get("experiment_hash") != expected:
            raise ValueError("Existing attribution run has a different experiment hash")
    else:
        manifest = _make_manifest(config, source_run_dir, source_manifest, run_dir)
        atomic_write_json(path, manifest)
    return manifest, source_manifest


def _completed(path: Path, digest: str, scenario_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = read_json(path)
    if (
        value.get("status") == "complete"
        and value.get("experiment_hash") == digest
        and value.get("scenario_hash") == scenario_hash
    ):
        return value
    return None


def _write_stage_jsonl(
    run_dir: Path, stage: str, filename: str
) -> list[dict[str, Any]]:
    directory = run_dir / "checkpoints" / stage
    records = [read_json(path) for path in sorted(directory.glob("*.json"))]
    completed = [item for item in records if item.get("status") == "complete"]
    write_jsonl(run_dir / filename, completed)
    return completed


def run_labeling(
    config: EvalConfig,
    source_run_dir: Path,
    run_dir: Path,
    *,
    limit: int | None = None,
    upload: bool = True,
    judge: Any | None = None,
) -> list[dict[str, Any]]:
    _, source_manifest = ensure_manifest(config, source_run_dir, run_dir)
    digest = experiment_hash(config, source_manifest)
    _, records, scenarios = validate_source_run(config, source_run_dir)
    selected = records[:limit] if limit else records
    checkpoint_dir = run_dir / "checkpoints" / "labels"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    store = _store(config, run_dir, upload)
    if store:
        store.upload_file(run_dir / "manifest.json", "manifest.json")
    labeler = judge or AttributionJudge(config.judge)
    for record in selected:
        checkpoint = checkpoint_dir / checkpoint_name(record["index"], record["uid"])
        if _completed(checkpoint, digest, record["scenario_hash"]):
            continue
        scenario = scenarios[record["uid"]]
        started = time.perf_counter()
        try:
            discoveries = {
                "old": discover_state_candidates(scenario["M_old"], scenario, record),
                "new": discover_state_candidates(scenario["M_new"], scenario, record),
            }
            labeled = labeler.label(scenario, discoveries)
            result = {
                "status": "complete",
                "experiment_hash": digest,
                "scenario_hash": record["scenario_hash"],
                "index": record["index"],
                "uid": record["uid"],
                "type": record["type"],
                "discoveries": discoveries,
                "ground_truth": labeled["result"],
                "labeler": {
                    key: value for key, value in labeled.items() if key != "result"
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            result = {
                "status": "error",
                "experiment_hash": digest,
                "scenario_hash": record["scenario_hash"],
                "index": record["index"],
                "uid": record["uid"],
                "type": record["type"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        atomic_write_json(checkpoint, result)
        if store:
            store.upload_file(checkpoint, str(checkpoint.relative_to(run_dir)))
        if result["status"] == "error":
            raise RuntimeError(
                f"Evidence labeling failed for {record['uid']}: {result['error']}"
            )
    return _write_stage_jsonl(run_dir, "labels", "evidence_labels.jsonl")


def _generate_cited_answer(
    qwen: Any,
    messages: list[dict[str, str]],
    allowed_ids: set[str],
    max_tokens: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    active_messages = list(messages)
    for parse_attempt in range(1, 4):
        generation = qwen.generate(
            active_messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        generation_data = asdict(generation)
        try:
            parsed = parse_citation_response(generation.content, allowed_ids)
            attempts.append({"status": "valid", "generation": generation_data})
            return {"parsed": parsed, "attempts": attempts}
        except Exception as exc:
            attempts.append(
                {
                    "status": "invalid",
                    "error": str(exc),
                    "generation": generation_data,
                }
            )
            active_messages.extend(
                [
                    {"role": "assistant", "content": generation.content},
                    {
                        "role": "user",
                        "content": (
                            f"Your JSON was invalid: {exc}. Return corrected JSON only, "
                            "with exactly four unique IDs copied from the evidence."
                        ),
                    },
                ]
            )
    raise RuntimeError(
        "Qwen failed to return a valid cited answer after three attempts"
    )


def run_cited_generation(
    config: EvalConfig,
    source_run_dir: Path,
    run_dir: Path,
    *,
    upload: bool = True,
    qwen: Any | None = None,
) -> list[dict[str, Any]]:
    _, source_manifest = ensure_manifest(config, source_run_dir, run_dir)
    digest = experiment_hash(config, source_manifest)
    _, source_records, _ = validate_source_run(config, source_run_dir)
    source_map = {item["uid"]: item for item in source_records}
    labels = _write_stage_jsonl(run_dir, "labels", "evidence_labels.jsonl")
    if not labels:
        raise RuntimeError("No evidence labels found; run the label stage first")
    checkpoint_dir = run_dir / "checkpoints" / "generation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    store = _store(config, run_dir, upload)
    client = qwen or NoThinkingQwenClient(config.qwen)
    for label in labels:
        source = source_map[label["uid"]]
        checkpoint = checkpoint_dir / checkpoint_name(label["index"], label["uid"])
        if _completed(checkpoint, digest, source["scenario_hash"]):
            continue
        started = time.perf_counter()
        try:
            old_ids = set(label["ground_truth"]["old"]["matching_ids"])
            new_ids = set(label["ground_truth"]["new"]["matching_ids"])
            answers: dict[str, Any] = {}
            for dim_key, dimension, _ in DIMENSIONS:
                original = source["answers"][dim_key]
                retrieval = original["retrieval"]
                messages = build_citation_prompt(
                    original["query"], retrieval, dimension, citation_count=4
                )
                allowed_ids = {str(item["id"]) for item in retrieval}
                generated = _generate_cited_answer(
                    client,
                    messages,
                    allowed_ids,
                    max_tokens=max(config.qwen.answer_max_tokens, 768),
                )
                parsed = generated["parsed"]
                answers[dim_key] = {
                    "dimension": dimension,
                    "query": original["query"],
                    "retrieval": retrieval,
                    "prompt": messages,
                    "answer": parsed["answer"],
                    "memory_ids_used": parsed["memory_ids_used"],
                    "citation_metrics": citation_metrics(
                        parsed["memory_ids_used"], retrieval, old_ids, new_ids
                    ),
                    "generation_attempts": generated["attempts"],
                }
            result = {
                "status": "complete",
                "experiment_hash": digest,
                "scenario_hash": source["scenario_hash"],
                "index": source["index"],
                "uid": source["uid"],
                "type": source["type"],
                "answers": answers,
                "elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            result = {
                "status": "error",
                "experiment_hash": digest,
                "scenario_hash": source["scenario_hash"],
                "index": source["index"],
                "uid": source["uid"],
                "type": source["type"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        atomic_write_json(checkpoint, result)
        if store:
            store.upload_file(checkpoint, str(checkpoint.relative_to(run_dir)))
        if result["status"] == "error":
            raise RuntimeError(
                f"Cited generation failed for {source['uid']}: {result['error']}"
            )
    return _write_stage_jsonl(run_dir, "generation", "cited_answers.jsonl")


def run_attribution_judging(
    config: EvalConfig,
    source_run_dir: Path,
    run_dir: Path,
    *,
    upload: bool = True,
    judge: Any | None = None,
) -> list[dict[str, Any]]:
    _, source_manifest = ensure_manifest(config, source_run_dir, run_dir)
    digest = experiment_hash(config, source_manifest)
    _, source_records, scenarios = validate_source_run(config, source_run_dir)
    source_map = {item["uid"]: item for item in source_records}
    labels = {
        item["uid"]: item
        for item in _write_stage_jsonl(run_dir, "labels", "evidence_labels.jsonl")
    }
    generations = _write_stage_jsonl(run_dir, "generation", "cited_answers.jsonl")
    if not generations:
        raise RuntimeError("No cited answers found; run generation first")
    checkpoint_dir = run_dir / "checkpoints" / "judge"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    store = _store(config, run_dir, upload)
    evaluator = judge or AttributionJudge(config.judge)
    for generation in generations:
        source = source_map[generation["uid"]]
        checkpoint = checkpoint_dir / checkpoint_name(
            generation["index"], generation["uid"]
        )
        if _completed(checkpoint, digest, source["scenario_hash"]):
            continue
        started = time.perf_counter()
        try:
            judged = evaluator.evaluate(
                scenarios[generation["uid"]],
                generation["answers"],
                labels[generation["uid"]]["ground_truth"],
            )
            result = {
                "status": "complete",
                "experiment_hash": digest,
                "scenario_hash": source["scenario_hash"],
                "index": source["index"],
                "uid": source["uid"],
                "type": source["type"],
                "evaluation": judged["result"],
                "judge": {
                    key: value for key, value in judged.items() if key != "result"
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            result = {
                "status": "error",
                "experiment_hash": digest,
                "scenario_hash": source["scenario_hash"],
                "index": source["index"],
                "uid": source["uid"],
                "type": source["type"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        atomic_write_json(checkpoint, result)
        if store:
            store.upload_file(checkpoint, str(checkpoint.relative_to(run_dir)))
        if result["status"] == "error":
            raise RuntimeError(
                f"Attribution judging failed for {source['uid']}: {result['error']}"
            )
    return _write_stage_jsonl(run_dir, "judge", "attribution_judgments.jsonl")


def build_attribution_summary(
    labels: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    generation_map = {item["uid"]: item for item in generations}
    answer_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    pattern_counts: Counter[str] = Counter()
    pattern_pass: dict[str, Counter[str]] = defaultdict(Counter)
    availability = Counter()
    usage = Counter()
    for judgment in judgments:
        generation = generation_map[judgment["uid"]]
        for dim_key, dim_name, _ in DIMENSIONS:
            answer = generation["answers"][dim_key]
            metrics = answer["citation_metrics"]
            passed = judgment["evaluation"][f"{dim_key}_eval"]["answer_pass"]
            for bucket in (
                "overall",
                dim_name,
                judgment["type"],
                f"{judgment['type']}×{dim_name}",
            ):
                answer_buckets[bucket]["total"] += 1
                answer_buckets[bucket]["passed"] += int(passed)
            pattern = metrics["evidence_pattern"]
            pattern_counts[pattern] += 1
            pattern_pass[pattern]["total"] += 1
            pattern_pass[pattern]["passed"] += int(passed)
            availability["old_available"] += int(metrics["old_ground_truth_available"])
            availability["new_available"] += int(metrics["new_ground_truth_available"])
            availability["old_used"] += int(metrics["used_old_evidence"])
            availability["new_used"] += int(metrics["used_new_evidence"])
            availability["questions"] += 1
            for attempt in answer.get("generation_attempts", []):
                generation_usage = attempt.get("generation", {}).get("usage", {})
                usage["qwen_tokens"] += generation_usage.get("total_tokens", 0)
        usage["judge_tokens"] += (
            judgment.get("judge", {}).get("usage", {}).get("total_tokens", 0)
        )
    for label in labels:
        usage["labeler_tokens"] += (
            label.get("labeler", {}).get("usage", {}).get("total_tokens", 0)
        )

    ordered_buckets = [
        "overall",
        "T1",
        "T2",
        "SR",
        "PR",
        "IPA",
        "T1×SR",
        "T1×PR",
        "T1×IPA",
        "T2×SR",
        "T2×PR",
        "T2×IPA",
    ]
    accuracy: dict[str, Any] = {}
    for key in ordered_buckets:
        bucket = answer_buckets[key]
        accuracy[key] = {
            "passed": bucket["passed"],
            "total": bucket["total"],
            "accuracy": bucket["passed"] / bucket["total"] if bucket["total"] else None,
        }
    patterns: dict[str, Any] = {}
    for key in ("old_only", "new_only", "both", "neither"):
        bucket = pattern_pass[key]
        patterns[key] = {
            "questions": pattern_counts[key],
            "answer_passed": bucket["passed"],
            "answer_accuracy": bucket["passed"] / bucket["total"]
            if bucket["total"]
            else None,
        }
    label_status = Counter(
        discovery.get("status", "unknown")
        for label in labels
        for discovery in label.get("discoveries", {}).values()
    )
    return {
        "status": "complete" if len(judgments) == 50 else "partial",
        "labeled_scenarios": len(labels),
        "generated_scenarios": len(generations),
        "judged_scenarios": len(judgments),
        "judged_questions": len(judgments) * 3,
        "answer_accuracy": accuracy,
        "evidence_patterns": patterns,
        "evidence_availability_and_use": dict(availability),
        "label_discovery_status": dict(label_status),
        "usage": dict(usage),
    }


def _escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def _render_report(
    summary: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
    source_map: dict[str, dict[str, Any]],
    labels: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> str:
    generation_map = {item["uid"]: item for item in generations}
    judgment_map = {item["uid"]: item for item in judgments}
    lines = [
        "# STALE evidence-attribution experiment",
        "",
        f"Status: **{summary['status']}**",
        "",
        "Qwen3-4B answered each question and cited exactly four retrieved memory IDs. Ground-truth M_old/M_new IDs were first narrowed from source-attributed chunks by gpt-6-luna. UUID overlap below is computed exactly; gpt-6-luna grades answer quality and interprets that verified evidence use.",
        "",
        "> Methodological caveat: the four IDs are Qwen's self-reported evidence attribution. They show which memories the model says influenced its answer, not causal proof that its output depended on those memories. A causal follow-up would rerun answers after masking or removing selected memories.",
        "",
        "## Answer accuracy",
        "",
        "| Section | Passed | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for key, value in summary["answer_accuracy"].items():
        accuracy = (
            "n/a" if value["accuracy"] is None else f"{100 * value['accuracy']:.2f}%"
        )
        lines.append(f"| {key} | {value['passed']} | {value['total']} | {accuracy} |")
    lines.extend(
        [
            "",
            "## Answer accuracy by cited evidence pattern",
            "",
            "| Pattern | Questions | Answers passed | Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, value in summary["evidence_patterns"].items():
        accuracy = (
            "n/a"
            if value["answer_accuracy"] is None
            else f"{100 * value['answer_accuracy']:.2f}%"
        )
        lines.append(
            f"| {key} | {value['questions']} | {value['answer_passed']} | {accuracy} |"
        )
    lines.extend(["", "## Per-scenario evidence", ""])

    for label in sorted(labels, key=lambda item: item["index"]):
        uid = label["uid"]
        scenario = scenarios[uid]
        generation = generation_map.get(uid)
        judgment = judgment_map.get(uid)
        lines.extend(
            [
                f'<a id="scenario-{label["index"]:02d}"></a>',
                f"## Scenario {label['index']:02d}: {label['type']} — {scenario['type_name']}",
                "",
                f"- UID: `{uid}`",
                f"- M_old: {scenario['M_old']}",
                f"- M_new: {scenario['M_new']}",
                f"- Hidden logic: {scenario['explanation']}",
                "",
                "### Ground-truth memory-ID narrowing",
                "",
            ]
        )
        for state_key, state_name in (("old", "M_old"), ("new", "M_new")):
            discovery = label["discoveries"][state_key]
            ground = label["ground_truth"][state_key]
            decisions = {item["id"]: item for item in ground["decisions"]}
            lines.extend(
                [
                    f"#### {state_name}",
                    "",
                    f"- Discovery status: `{discovery['status']}`",
                    f"- Verified matching IDs: `{', '.join(ground['matching_ids']) or '(none)'}`",
                    "",
                ]
            )
            source = discovery.get("source")
            if source:
                lines.extend(["**Source message:**", "", f"> {source['content']}", ""])
            candidates = discovery.get("eligible_candidates", [])
            if candidates:
                lines.extend(
                    [
                        "| ID | Post-reconciliation memory | Label | Labeler reasoning |",
                        "|---|---|---|---|",
                    ]
                )
                for candidate in candidates:
                    decision = decisions.get(candidate["id"], {})
                    lines.append(
                        "| `{}` | {} | {} | {} |".format(
                            candidate["id"],
                            _escape(candidate["post_memory"]),
                            "MATCH" if decision.get("matches") else "exclude",
                            _escape(decision.get("reasoning", "")),
                        )
                    )
                lines.append("")
            else:
                lines.extend(["_No retrievable source-attributed candidates._", ""])

        if generation is None:
            lines.extend(["_No cited-answer generation record._", "", "---", ""])
            continue
        for dim_key, dim_name, long_name in DIMENSIONS:
            answer = generation["answers"][dim_key]
            evaluation = (
                judgment.get("evaluation", {}).get(f"{dim_key}_eval", {})
                if judgment
                else {}
            )
            metrics = answer["citation_metrics"]
            old_set = set(metrics["old_ground_truth_ids"])
            new_set = set(metrics["new_ground_truth_ids"])
            lines.extend(
                [
                    f"### {dim_name}: {long_name}",
                    "",
                    f"**Question:** {answer['query']}",
                    "",
                    f"- Evidence pattern: **{metrics['evidence_pattern']}**",
                    f"- M_old available in retrieved top 20: **{metrics['old_ground_truth_available']}**",
                    f"- M_new available in retrieved top 20: **{metrics['new_ground_truth_available']}**",
                    f"- Selected M_old IDs: `{', '.join(metrics['selected_old_ids']) or '(none)'}`",
                    f"- Selected M_new IDs: `{', '.join(metrics['selected_new_ids']) or '(none)'}`",
                    "",
                    "#### Retrieved evidence",
                    "",
                    "| Rank | Score | ID | State label | Memory |",
                    "|---:|---:|---|---|---|",
                ]
            )
            for rank, item in enumerate(answer["retrieval"], start=1):
                memory_id = str(item["id"])
                if memory_id in old_set and memory_id in new_set:
                    state_label = "M_old + M_new"
                elif memory_id in old_set:
                    state_label = "M_old"
                elif memory_id in new_set:
                    state_label = "M_new"
                else:
                    state_label = "other"
                lines.append(
                    "| {} | {:.6f} | `{}` | {} | {} |".format(
                        rank,
                        float(item.get("score", 0.0)),
                        memory_id,
                        state_label,
                        _escape(item.get("payload", {}).get("memory", "")),
                    )
                )
            retrieval_by_id = {str(item["id"]): item for item in answer["retrieval"]}
            lines.extend(
                [
                    "",
                    "#### Qwen's four cited memories",
                    "",
                    "| Importance | ID | State label | Memory |",
                    "|---:|---|---|---|",
                ]
            )
            for rank, memory_id in enumerate(answer["memory_ids_used"], start=1):
                item = retrieval_by_id[memory_id]
                if memory_id in old_set and memory_id in new_set:
                    state_label = "M_old + M_new"
                elif memory_id in old_set:
                    state_label = "M_old"
                elif memory_id in new_set:
                    state_label = "M_new"
                else:
                    state_label = "other"
                lines.append(
                    f"| {rank} | `{memory_id}` | {state_label} | "
                    f"{_escape(item.get('payload', {}).get('memory', ''))} |"
                )
            lines.extend(
                [
                    "",
                    "#### Qwen answer",
                    "",
                    answer["answer"],
                    "",
                ]
            )
            if evaluation:
                lines.extend(
                    [
                        f"**Answer judge:** {'PASS' if evaluation['answer_pass'] else 'FAIL'}",
                        "",
                        f"**Answer reasoning:** {evaluation['answer_reasoning']}",
                        "",
                        f"**M_old evidence assessment:** {evaluation['old_evidence']['reasoning']}",
                        "",
                        f"**M_new evidence assessment:** {evaluation['new_evidence']['reasoning']}",
                        "",
                        f"**Citation assessment:** {evaluation['citation_assessment']}",
                        "",
                    ]
                )
            lines.extend(
                [
                    "<details><summary>Exact Qwen prompt</summary>",
                    "",
                    "````json",
                    json.dumps(answer["prompt"], ensure_ascii=False, indent=2),
                    "````",
                    "",
                    "</details>",
                    "",
                ]
            )
        lines.extend(["---", ""])
    return "\n".join(lines).rstrip() + "\n"


def generate_attribution_report(
    config: EvalConfig,
    source_run_dir: Path,
    run_dir: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    manifest, _ = ensure_manifest(config, source_run_dir, run_dir)
    _, source_records, scenarios = validate_source_run(config, source_run_dir)
    source_map = {item["uid"]: item for item in source_records}
    labels = _write_stage_jsonl(run_dir, "labels", "evidence_labels.jsonl")
    generations = _write_stage_jsonl(run_dir, "generation", "cited_answers.jsonl")
    judgments = _write_stage_jsonl(run_dir, "judge", "attribution_judgments.jsonl")
    summary = build_attribution_summary(labels, generations, judgments)
    atomic_write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(
        _render_report(summary, scenarios, source_map, labels, generations, judgments),
        encoding="utf-8",
    )
    manifest["status"] = summary["status"]
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(run_dir / "manifest.json", manifest)
    if summary["status"] != "complete" and not allow_partial:
        raise RuntimeError(
            f"Attribution run is incomplete: judged {len(judgments)} of 50 scenarios"
        )
    return summary


def upload_attribution_run(config: EvalConfig, run_dir: Path) -> list[str]:
    if not config.s3.enabled:
        return []
    return S3ArtifactStore(evidence_s3_config(config), run_dir.name).upload_run(run_dir)
