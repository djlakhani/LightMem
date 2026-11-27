from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from .artifacts import (
    atomic_write_json,
    checkpoint_name,
    completed_checkpoint,
    read_json,
    write_jsonl,
)
from .attribution import parse_json_object
from .config import EvalConfig, JudgeConfig
from .dataset import DIMENSIONS, load_dataset
from .runner import load_run_manifest
from .s3 import S3ArtifactStore

TELEMETRY_SCHEMA_VERSION = 1
DIMENSION_LONG_NAMES = {
    "dim1": "State Recognition",
    "dim2": "Premise Resistance",
    "dim3": "Implicitly Personalized Action",
}

SEMANTIC_LABEL_SYSTEM_PROMPT = """
You are auditing semantic evidence recall in a long-term-memory evaluation.

You receive the benchmark's M_old, M_new, hidden update logic, and EVERY memory
that survived consolidation. Determine which memories preserve each state. UUID
or source-message identity is not evidence: judge the actual memory text.

A state can be captured by one memory or only by a combination of memories.
Classify every memory for whether it independently explicitly states, clearly
paraphrases, or necessarily entails each state. Also list every minimal support
set (one or more IDs) whose combined contents capture the state. Do not select
mere topical overlap, downstream consequences, or facts requiring unsupported
inference. An empty support_sets array means the state is absent.

Return only JSON in this exact shape:
{
  "old": {
    "decisions": [{"id": "...", "matches": true, "reasoning": "..."}],
    "support_sets": [{"memory_ids": ["..."], "reasoning": "..."}]
  },
  "new": {
    "decisions": [{"id": "...", "matches": true, "reasoning": "..."}],
    "support_sets": [{"memory_ids": ["..."], "reasoning": "..."}]
  }
}
"""


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def semantic_label_prompt(
    scenario: dict[str, Any], memories: list[dict[str, Any]]
) -> str:
    sections = [
        "[Ground truth]",
        f"M_old: {scenario['M_old']}",
        f"M_new: {scenario['M_new']}",
        f"Hidden logic: {scenario['explanation']}",
        "",
        "[Complete post-consolidation memory inventory]",
    ]
    for memory in memories:
        payload = memory.get("payload", {})
        sections.extend(
            [
                f"ID: {memory['id']}",
                f"Timestamp: {payload.get('time_stamp', '')}",
                "Attributes: "
                + json.dumps(payload.get("attributes", []), ensure_ascii=False),
                f"Memory: {payload.get('memory', '')}",
                "",
            ]
        )
    return "\n".join(sections).rstrip()


def parse_semantic_labels(
    content: str, memories: list[dict[str, Any]]
) -> dict[str, Any]:
    value = parse_json_object(content)
    expected = {str(memory["id"]) for memory in memories}
    result: dict[str, Any] = {}
    for state in ("old", "new"):
        section = value.get(state)
        if not isinstance(section, dict) or set(section) != {
            "decisions",
            "support_sets",
        }:
            raise ValueError(
                f"Semantic labeler {state} must contain decisions and support_sets"
            )
        decisions = section["decisions"]
        if not isinstance(decisions, list):
            raise ValueError(f"Semantic labeler {state}.decisions must be a list")
        normalized_decisions: list[dict[str, Any]] = []
        observed: set[str] = set()
        for decision in decisions:
            if not isinstance(decision, dict) or set(decision) != {
                "id",
                "matches",
                "reasoning",
            }:
                raise ValueError(f"Malformed {state} memory decision")
            memory_id = decision["id"]
            if memory_id not in expected or memory_id in observed:
                raise ValueError(f"Invalid or duplicate {state} memory ID: {memory_id}")
            if not isinstance(decision["matches"], bool) or not isinstance(
                decision["reasoning"], str
            ):
                raise ValueError(f"Malformed {state} decision fields for {memory_id}")
            observed.add(memory_id)
            normalized_decisions.append(decision)
        if observed != expected:
            raise ValueError(
                f"Semantic labeler did not classify every memory for {state}; "
                f"missing={sorted(expected - observed)}"
            )

        support_sets = section["support_sets"]
        if not isinstance(support_sets, list):
            raise ValueError(f"Semantic labeler {state}.support_sets must be a list")
        normalized_sets: list[dict[str, Any]] = []
        seen_sets: set[tuple[str, ...]] = set()
        for support in support_sets:
            if not isinstance(support, dict) or set(support) != {
                "memory_ids",
                "reasoning",
            }:
                raise ValueError(f"Malformed {state} support set")
            ids = support["memory_ids"]
            reasoning = support["reasoning"]
            if (
                not isinstance(ids, list)
                or not ids
                or not all(isinstance(item, str) for item in ids)
                or len(ids) != len(set(ids))
                or not set(ids) <= expected
                or not isinstance(reasoning, str)
                or not reasoning.strip()
            ):
                raise ValueError(f"Invalid {state} support set: {support}")
            canonical = tuple(sorted(ids))
            if canonical in seen_sets:
                raise ValueError(f"Duplicate {state} support set: {canonical}")
            seen_sets.add(canonical)
            normalized_sets.append(
                {"memory_ids": list(canonical), "reasoning": reasoning.strip()}
            )

        # An independently matching memory is necessarily a singleton support set.
        for decision in normalized_decisions:
            if decision["matches"]:
                canonical = (decision["id"],)
                if canonical not in seen_sets:
                    seen_sets.add(canonical)
                    normalized_sets.append(
                        {
                            "memory_ids": [decision["id"]],
                            "reasoning": decision["reasoning"].strip(),
                        }
                    )
        participating_ids = sorted(
            {item for support in normalized_sets for item in support["memory_ids"]}
        )
        result[state] = {
            "decisions": normalized_decisions,
            "support_sets": normalized_sets,
            "independent_matching_ids": sorted(
                item["id"] for item in normalized_decisions if item["matches"]
            ),
            "participating_ids": participating_ids,
        }
    return result


class SemanticStateJudge:
    def __init__(self, config: JudgeConfig):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for semantic telemetry")
        self.config = config
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))

    def label(
        self, scenario: dict[str, Any], memories: list[dict[str, Any]]
    ) -> dict[str, Any]:
        user_prompt = semantic_label_prompt(scenario, memories)
        last_error: Exception | None = None
        attempt_log: list[dict[str, Any]] = []
        aggregate_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        started_all = time.perf_counter()
        for attempt in range(1, self.config.max_retries + 1):
            started = time.perf_counter()
            content = ""
            response_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": SEMANTIC_LABEL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                response_usage = _usage(response)
                for key in aggregate_usage:
                    aggregate_usage[key] += response_usage[key]
                parsed = parse_semantic_labels(content, memories)
                attempt_log.append(
                    {
                        "attempt": attempt,
                        "status": "valid",
                        "elapsed_seconds": time.perf_counter() - started,
                        "raw_response": content,
                        "usage": response_usage,
                    }
                )
                return {
                    "result": parsed,
                    "raw_response": content,
                    "attempts": attempt,
                    "attempt_log": attempt_log,
                    "elapsed_seconds": time.perf_counter() - started_all,
                    "usage": aggregate_usage,
                    "model": self.config.model,
                    "reasoning_enabled": self.config.reasoning_enabled,
                    "prompt": {
                        "system": SEMANTIC_LABEL_SYSTEM_PROMPT,
                        "user": user_prompt,
                    },
                }
            except Exception as exc:
                last_error = exc
                attempt_log.append(
                    {
                        "attempt": attempt,
                        "status": "invalid",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": time.perf_counter() - started,
                        "raw_response": content,
                        "usage": response_usage,
                    }
                )
                if attempt < self.config.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(
            f"Semantic labeler failed after {self.config.max_retries} attempts: "
            f"{attempt_log}"
        ) from last_error


def state_presence(ids: list[str], state: dict[str, Any]) -> dict[str, Any]:
    available = set(ids)
    satisfied = [
        support
        for support in state.get("support_sets", [])
        if set(support["memory_ids"]) <= available
    ]
    partial = []
    for support in state.get("support_sets", []):
        overlap = sorted(set(support["memory_ids"]) & available)
        if overlap and support not in satisfied:
            partial.append(
                {
                    "memory_ids": support["memory_ids"],
                    "present_ids": overlap,
                    "missing_ids": sorted(set(support["memory_ids"]) - available),
                }
            )
    return {
        "present": bool(satisfied),
        "satisfied_support_sets": satisfied,
        "partial_support_sets": partial,
        "participating_ids_present": sorted(
            available & set(state.get("participating_ids", []))
        ),
    }


def question_stages(answer: dict[str, Any]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []

    def add(name: str, label: str, memories: list[dict[str, Any]]) -> None:
        stages.append(
            {
                "name": name,
                "label": label,
                "memory_ids": [str(item["id"]) for item in memories],
            }
        )

    add("top20", "Embedding retrieval (top 20)", answer.get("retrieval", []))
    anchors = answer.get("evidence_selection", {}).get("selected_memories", [])
    add("anchors", "Qwen-selected anchors", anchors)
    contexts = answer.get("future_update_analysis", {}).get("anchors", [])
    all_candidates: list[dict[str, Any]] = []
    all_updates: list[dict[str, Any]] = []
    for index, context in enumerate(contexts, start=1):
        candidates = context.get("candidate_search", {}).get("candidates", [])
        updates = context.get("selected_updates", [])
        all_candidates.extend(candidates)
        all_updates.extend(updates)
        add(
            f"anchor_{index}_candidates",
            f"Anchor {index} matched future candidates",
            candidates,
        )
        add(
            f"anchor_{index}_updates",
            f"Anchor {index} retained future updates",
            updates,
        )
    add(
        "all_future_candidates",
        "All matched future candidates",
        _unique(all_candidates),
    )
    add("all_selected_updates", "All retained future updates", _unique(all_updates))
    add(
        "final_prompt_evidence", "Final prompt evidence", _unique(anchors + all_updates)
    )
    return stages


def _unique(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in memories:
        memory_id = str(item["id"])
        if memory_id not in seen:
            seen.add(memory_id)
            result.append(item)
    return result


def annotate_question(
    answer: dict[str, Any], semantic_labels: dict[str, Any]
) -> dict[str, Any]:
    stages = question_stages(answer)
    for stage in stages:
        stage["old"] = state_presence(stage["memory_ids"], semantic_labels["old"])
        stage["new"] = state_presence(stage["memory_ids"], semantic_labels["new"])
        stage["old_and_new"] = stage["old"]["present"] and stage["new"]["present"]
    return {"stages": stages}


def _store(config: EvalConfig, run_dir: Path, upload: bool) -> S3ArtifactStore | None:
    if upload and config.s3.enabled:
        return S3ArtifactStore(config.s3, run_dir.name)
    return None


def run_telemetry_analysis(
    config: EvalConfig,
    run_dir: Path,
    *,
    upload: bool = True,
    judge: Any | None = None,
) -> list[dict[str, Any]]:
    manifest = load_run_manifest(run_dir)
    if manifest.get("config_hash") != config.digest:
        raise ValueError("Run config hash does not match telemetry configuration")
    dataset = load_dataset(config.resolved_scenario_path, config.scenario_sha256)
    scenarios = {scenario["uid"]: scenario for scenario in dataset.scenarios}
    generation_dir = run_dir / "checkpoints" / "generation"
    records = [read_json(path) for path in sorted(generation_dir.glob("*.json"))]
    records = [record for record in records if record.get("status") == "complete"]
    if not records:
        raise RuntimeError("No completed generation checkpoints were found")

    checkpoint_dir = run_dir / "checkpoints" / "telemetry"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    store = _store(config, run_dir, upload)
    labeler = judge or SemanticStateJudge(config.judge)
    for record in records:
        checkpoint = checkpoint_dir / checkpoint_name(record["index"], record["uid"])
        existing = completed_checkpoint(
            checkpoint, config.digest, record["scenario_hash"]
        )
        if existing and existing.get("schema_version") == TELEMETRY_SCHEMA_VERSION:
            continue
        started = time.perf_counter()
        try:
            labeled = labeler.label(
                scenarios[record["uid"]], record["post_update_memories"]
            )
            result = {
                "status": "complete",
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "config_hash": config.digest,
                "scenario_hash": record["scenario_hash"],
                "index": record["index"],
                "uid": record["uid"],
                "type": record["type"],
                "semantic_labels": labeled["result"],
                "questions": {
                    key: annotate_question(answer, labeled["result"])
                    for key, answer in record["answers"].items()
                },
                "labeler": {
                    key: value for key, value in labeled.items() if key != "result"
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            result = {
                "status": "error",
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "config_hash": config.digest,
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
                f"Telemetry labeling failed for {record['uid']}: {result['error']}"
            )

    telemetry = [read_json(path) for path in sorted(checkpoint_dir.glob("*.json"))]
    completed = [record for record in telemetry if record.get("status") == "complete"]
    output = run_dir / "telemetry_records.jsonl"
    write_jsonl(output, completed)
    if store:
        store.upload_file(output, output.name)
    return completed


def _fallback_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        attributes = record.get("token_statistics", {}).get("attributes", {})
        for key, value in attributes.items():
            counts[f"attribute_{key}"] += int(value or 0)
        for answer in record.get("answers", {}).values():
            selection = answer.get("evidence_selection", {}).get("generation", {})
            counts["anchor_questions"] += 1
            counts["anchor_repair_attempts"] += int(
                selection.get("llm_attempts", 0) > 1
            )
            counts["anchor_embedding_fallbacks"] += int(
                bool(selection.get("fallback_used"))
            )
            counts["answer_transport_retries"] += max(
                0, int(answer.get("generation", {}).get("attempts", 1)) - 1
            )
            for anchor in answer.get("future_update_analysis", {}).get("anchors", []):
                result = anchor.get("filter_result", {})
                counts["future_filters"] += 1
                counts["future_no_candidates"] += int(
                    result.get("method") == "no_candidates"
                )
                counts["future_repair_attempts"] += int(
                    result.get("llm_attempts", 0) > 1
                )
                counts["future_no_update_fallbacks"] += int(
                    bool(result.get("fallback_used"))
                )
    return dict(sorted(counts.items()))


def build_telemetry_summary(
    records: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_counts: dict[str, Counter[str]] = {}
    telemetry_map = {item["uid"]: item for item in telemetry}
    for record in records:
        analyzed = telemetry_map.get(record["uid"], {})
        for question in analyzed.get("questions", {}).values():
            for stage in question.get("stages", []):
                counter = stage_counts.setdefault(stage["name"], Counter())
                counter["questions"] += 1
                counter["old"] += int(stage["old"]["present"])
                counter["new"] += int(stage["new"]["present"])
                counter["old_and_new"] += int(stage["old_and_new"])
    passed = Counter()
    relationships = Counter()
    for record in records:
        for answer in record.get("answers", {}).values():
            for anchor in answer.get("future_update_analysis", {}).get("anchors", []):
                for update in anchor.get("selected_updates", []):
                    relationships[
                        update.get("update_relationship", "unclassified")
                    ] += 1
    for judgment in judgments:
        for dim_key, _, _ in DIMENSIONS:
            passed["questions"] += 1
            passed["passed"] += int(
                judgment.get("evaluation", {})
                .get(f"{dim_key}_eval", {})
                .get("pass", False)
            )
    return {
        "status": (
            "complete"
            if records
            and len(telemetry) == len(records)
            and len(judgments) == len(records)
            else "partial"
        ),
        "scenarios": len(records),
        "questions": len(records) * 3,
        "telemetry_scenarios": len(telemetry),
        "judged_scenarios": len(judgments),
        "answer_accuracy": (
            passed["passed"] / passed["questions"] if passed["questions"] else None
        ),
        "answer_passed": passed["passed"],
        "stage_presence": {
            key: dict(value) for key, value in sorted(stage_counts.items())
        },
        "fallbacks": _fallback_counts(records),
        "selected_update_relationships": dict(sorted(relationships.items())),
        "telemetry_labeler": {
            "calls": len(telemetry),
            "retried_calls": sum(
                item.get("labeler", {}).get("attempts", 1) > 1 for item in telemetry
            ),
            "tokens": sum(
                item.get("labeler", {}).get("usage", {}).get("total_tokens", 0)
                for item in telemetry
            ),
            "seconds": sum(item.get("elapsed_seconds", 0.0) for item in telemetry),
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _state_mark(memory_id: str, labels: dict[str, Any]) -> str:
    old = memory_id in set(labels["old"]["participating_ids"])
    new = memory_id in set(labels["new"]["participating_ids"])
    if old and new:
        return "M_old + M_new"
    if old:
        return "M_old"
    if new:
        return "M_new"
    return "other"


def _memory_table(
    memories: list[dict[str, Any]],
    labels: dict[str, Any],
    *,
    selected_ids: set[str] | None = None,
) -> list[str]:
    lines = [
        "| # | ID | Score | Timestamp | Attributes | State | Selection | Memory |",
        "|---:|---|---:|---|---|---|---|---|",
    ]
    if not memories:
        return ["_(No memories.)_", ""]
    for rank, item in enumerate(memories, start=1):
        memory_id = str(item["id"])
        payload = item.get("payload", {})
        score = item.get("score")
        score_text = "" if score is None else f"{float(score):.6f}"
        selection = ""
        if selected_ids is not None:
            provenance = (
                f"source={item.get('candidate_source', '')}; "
                f"matched={item.get('matched_attributes', [])}"
            )
            if memory_id in selected_ids:
                relationship = item.get("update_relationship", "selected")
                explanation = item.get("update_explanation", "")
                resolution = f"{relationship}: {explanation}".rstrip(": ")
            else:
                resolution = "rejected"
            selection = f"{provenance}; {resolution}"
        lines.append(
            "| {} | `{}` | {} | {} | {} | {} | {} | {} |".format(
                rank,
                memory_id,
                score_text,
                _escape(payload.get("time_stamp", "")),
                _escape(json.dumps(payload.get("attributes", []), ensure_ascii=False)),
                _state_mark(memory_id, labels),
                _escape(selection),
                _escape(payload.get("memory", "")),
            )
        )
    lines.append("")
    return lines


def _format_support_sets(state: dict[str, Any]) -> str:
    if not state.get("support_sets"):
        return "(absent)"
    return "; ".join(
        "{" + ", ".join(item["memory_ids"]) + "}" for item in state["support_sets"]
    )


def _render_fallbacks(
    record: dict[str, Any],
    answer: dict[str, Any],
    analyzed: dict[str, Any],
    judgment: dict[str, Any],
) -> list[str]:
    selection = answer.get("evidence_selection", {}).get("generation", {})
    lines = [
        f"- Attribute statistics (scenario): `{json.dumps(record.get('token_statistics', {}).get('attributes', {}), sort_keys=True)}`",
        f"- Anchor selection: method=`{selection.get('method')}`, attempts={selection.get('llm_attempts')}, fallback={selection.get('fallback_used')}, reason={selection.get('fallback_reason')}",
        f"- Answer call attempts: {answer.get('generation', {}).get('attempts', 0)}",
        f"- Answer transport errors: `{json.dumps(answer.get('generation', {}).get('retry_errors', []), ensure_ascii=False)}`",
        f"- Semantic labeler attempts (scenario): {analyzed.get('labeler', {}).get('attempts', 0)}",
        f"- GPT answer-judge attempts (scenario): {judgment.get('attempts', 0)}",
    ]
    for index, anchor in enumerate(
        answer.get("future_update_analysis", {}).get("anchors", []), start=1
    ):
        result = anchor.get("filter_result", {})
        lines.append(
            f"- Anchor {index} future filter: method=`{result.get('method')}`, "
            f"attempts={result.get('llm_attempts')}, fallback={result.get('fallback_used')}, "
            f"reason={result.get('fallback_reason')}"
        )
    lines.append("")
    return lines


def render_telemetry_report(
    summary: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> str:
    telemetry_map = {item["uid"]: item for item in telemetry}
    judgment_map = {item["uid"]: item for item in judgments}
    lines = [
        "# STALE LightMem filtering and semantic-recall telemetry",
        "",
        f"Status: **{summary['status']}**",
        "",
        (
            f"Scenarios: **{summary['scenarios']}** · "
            f"Questions: **{summary['questions']}** · "
            f"Answer accuracy: **{summary['answer_passed']}/{summary['questions']}**"
        ),
        "",
        (
            "M_old/M_new support sets were labeled by gpt-6-luna over the complete "
            "post-consolidation memory inventory. Stage presence below is a "
            "deterministic set-containment check; it is not inferred from source IDs."
        ),
        "",
        "## Aggregate stage recall",
        "",
        "| Stage | Questions | M_old | M_new | Both |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, counts in summary["stage_presence"].items():
        lines.append(
            f"| {name} | {counts.get('questions', 0)} | {counts.get('old', 0)} | "
            f"{counts.get('new', 0)} | {counts.get('old_and_new', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate fallback ledger",
            "",
            "```json",
            json.dumps(summary["fallbacks"], indent=2, sort_keys=True),
            "```",
            "",
            "## Selected future-state relationships",
            "",
            "```json",
            json.dumps(
                summary["selected_update_relationships"], indent=2, sort_keys=True
            ),
            "```",
            "",
        ]
    )

    for record in sorted(records, key=lambda item: item["index"]):
        scenario = scenarios[record["uid"]]
        analyzed = telemetry_map.get(record["uid"])
        judgment = judgment_map.get(record["uid"], {})
        lines.extend(
            [
                f"## Scenario {record['index']:02d}: {record['type']} — {record['type_name']}",
                "",
                f"- UID: `{record['uid']}`",
                f"- Context view: `{record['context_view']}`",
                f"- M_old: {scenario['M_old']}",
                f"- M_new: {scenario['M_new']}",
                f"- Hidden logic: {scenario['explanation']}",
                f"- Memory diff: `{json.dumps(record['memory_diff'], sort_keys=True)}`",
                "",
            ]
        )
        if analyzed is None:
            lines.extend(["_Semantic telemetry missing._", ""])
            continue
        labels = analyzed["semantic_labels"]
        lines.extend(
            [
                "### Semantic memory ground truth",
                "",
                f"- M_old support sets: `{_format_support_sets(labels['old'])}`",
                f"- M_new support sets: `{_format_support_sets(labels['new'])}`",
                f"- Labeler attempts: {analyzed['labeler'].get('attempts', 0)}",
                "",
            ]
        )
        lines.extend(_memory_table(record["post_update_memories"], labels))
        for state_key, state_name in (("old", "M_old"), ("new", "M_new")):
            lines.extend(
                [
                    f"#### {state_name} support-set judgments",
                    "",
                    "| Support memory IDs | GPT-6-luna reasoning |",
                    "|---|---|",
                ]
            )
            support_sets = labels[state_key].get("support_sets", [])
            if support_sets:
                for support in support_sets:
                    lines.append(
                        f"| `{json.dumps(support['memory_ids'])}` | "
                        f"{_escape(support['reasoning'])} |"
                    )
            else:
                lines.append("| _(absent)_ | No surviving support set. |")
            lines.append("")
        lines.extend(
            [
                "<details><summary>All semantic memory decisions and exact labeler trace</summary>",
                "",
                "````json",
                json.dumps(
                    {
                        "semantic_labels": labels,
                        "labeler": analyzed.get("labeler", {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "````",
                "",
                "</details>",
                "",
            ]
        )

        for dim_key, short_name, _ in DIMENSIONS:
            long_name = DIMENSION_LONG_NAMES[dim_key]
            answer = record["answers"][dim_key]
            audit = analyzed["questions"][dim_key]
            evaluation = judgment.get("evaluation", {}).get(f"{dim_key}_eval", {})
            lines.extend(
                [
                    f"### {short_name}: {long_name}",
                    "",
                    f"**Question:** {answer['query']}",
                    "",
                    "#### Stage-by-stage semantic presence",
                    "",
                    "| Stage | Count | M_old | M_new | Both | Old support found | New support found |",
                    "|---|---:|---|---|---|---|---|",
                ]
            )
            for stage in audit["stages"]:
                old_sets = [
                    item["memory_ids"]
                    for item in stage["old"]["satisfied_support_sets"]
                ]
                new_sets = [
                    item["memory_ids"]
                    for item in stage["new"]["satisfied_support_sets"]
                ]
                lines.append(
                    f"| {stage['label']} | {len(stage['memory_ids'])} | "
                    f"{stage['old']['present']} | {stage['new']['present']} | "
                    f"{stage['old_and_new']} | `{json.dumps(old_sets)}` | "
                    f"`{json.dumps(new_sets)}` |"
                )

            stage_by_name = {stage["name"]: stage for stage in audit["stages"]}
            top20_stage = stage_by_name["top20"]
            anchor_stage = stage_by_name["anchors"]
            final_stage = stage_by_name["final_prompt_evidence"]
            lines.extend(
                [
                    "",
                    (
                        "**Key recall transitions:** "
                        f"top20(old={top20_stage['old']['present']}, "
                        f"new={top20_stage['new']['present']}) → "
                        f"anchors(old={anchor_stage['old']['present']}, "
                        f"new={anchor_stage['new']['present']}) → "
                        f"final(old={final_stage['old']['present']}, "
                        f"new={final_stage['new']['present']})."
                    ),
                    "",
                    "<details><summary>Exact anchor-selection prompt and attempts</summary>",
                    "",
                    "````json",
                    json.dumps(
                        {
                            "prompt": answer.get("evidence_selection", {}).get(
                                "prompt", []
                            ),
                            "result": answer.get("evidence_selection", {}).get(
                                "generation", {}
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "````",
                    "",
                    "</details>",
                    "",
                ]
            )

            lines.extend(["#### Top-20 embedding retrieval", ""])
            lines.extend(_memory_table(answer.get("retrieval", []), labels))
            anchors = answer.get("evidence_selection", {}).get("selected_memories", [])
            lines.extend(["#### Four Qwen anchors", ""])
            lines.extend(_memory_table(anchors, labels))

            for anchor_index, context in enumerate(
                answer.get("future_update_analysis", {}).get("anchors", []), start=1
            ):
                anchor = context["anchor"]
                candidates = context.get("candidate_search", {}).get("candidates", [])
                selected = context.get("selected_updates", [])
                selected_ids = {str(item["id"]) for item in selected}
                selected_by_id = {str(item["id"]): item for item in selected}
                resolved_candidates = [
                    {**candidate, **selected_by_id.get(str(candidate["id"]), {})}
                    for candidate in candidates
                ]
                lines.extend(
                    [
                        f"#### Anchor {anchor_index}: matched and resolved future states",
                        "",
                        "**Anchor:**",
                        "",
                    ]
                )
                lines.extend(_memory_table([anchor], labels))
                lines.extend(
                    [
                        (
                            "Eligible after time/attribute filtering: "
                            f"{context.get('candidate_search', {}).get('eligible_count', 0)}"
                        ),
                        "",
                    ]
                )
                lines.extend(
                    _memory_table(
                        resolved_candidates, labels, selected_ids=selected_ids
                    )
                )
                lines.extend(
                    [
                        "<details><summary>Exact future-filter prompt and attempts</summary>",
                        "",
                        "````json",
                        json.dumps(
                            {
                                "prompt": context.get("filter_prompt", []),
                                "result": context.get("filter_result", {}),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        "````",
                        "",
                        "</details>",
                        "",
                    ]
                )

            by_id = {str(item["id"]): item for item in record["post_update_memories"]}
            final_detail_by_id = {
                str(item["id"]): item
                for item in answer.get("evidence_selection", {}).get(
                    "selected_memories", []
                )
            }
            for context in answer.get("future_update_analysis", {}).get("anchors", []):
                for item in context.get("selected_updates", []):
                    final_detail_by_id[str(item["id"])] = item
            final_memories = [
                final_detail_by_id.get(memory_id, by_id[memory_id])
                for memory_id in final_stage["memory_ids"]
                if memory_id in by_id
            ]
            lines.extend(["#### Final answer evidence and exact prompt", ""])
            lines.extend(
                _memory_table(
                    final_memories,
                    labels,
                    selected_ids=set(final_stage["memory_ids"]),
                )
            )
            lines.extend(
                [
                    f"- Final prompt contains M_old: **{final_stage['old']['present']}**",
                    f"- Final prompt contains M_new: **{final_stage['new']['present']}**",
                    f"- Final prompt contains M_old + M_new: **{final_stage['old_and_new']}**",
                    f"- Final prompt characters: **{len(json.dumps(answer['prompt'], ensure_ascii=False))}**",
                    "",
                    "````json",
                    json.dumps(answer["prompt"], ensure_ascii=False, indent=2),
                    "````",
                    "",
                    "#### Qwen response and judge",
                    "",
                    answer["answer"],
                    "",
                    f"**Judge:** {'PASS' if evaluation.get('pass') else 'FAIL'}",
                    "",
                    f"**Judge reasoning:** {evaluation.get('reasoning', '(missing)')}",
                    "",
                    "#### Fallback and retry telemetry",
                    "",
                ]
            )
            lines.extend(_render_fallbacks(record, answer, analyzed, judgment))
        lines.extend(
            [
                "### Scenario timing and token telemetry",
                "",
                "```json",
                json.dumps(
                    {
                        "elapsed_seconds": record.get("elapsed_seconds"),
                        "ingest": record.get("ingest", {}).get("elapsed_seconds"),
                        "consolidation": record.get("consolidation", {}).get(
                            "elapsed_seconds"
                        ),
                        "token_statistics": record.get("token_statistics", {}),
                        "semantic_labeler": analyzed.get("labeler", {}),
                        "answer_judge": {
                            key: value
                            for key, value in judgment.items()
                            if key not in {"evaluation", "prompt"}
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def generate_telemetry_report(
    config: EvalConfig,
    run_dir: Path,
    *,
    allow_partial: bool = False,
    upload: bool = True,
) -> dict[str, Any]:
    dataset = load_dataset(config.resolved_scenario_path, config.scenario_sha256)
    scenarios = {scenario["uid"]: scenario for scenario in dataset.scenarios}
    records = [
        read_json(path)
        for path in sorted((run_dir / "checkpoints" / "generation").glob("*.json"))
    ]
    records = [record for record in records if record.get("status") == "complete"]
    telemetry = [
        read_json(path)
        for path in sorted((run_dir / "checkpoints" / "telemetry").glob("*.json"))
    ]
    telemetry = [item for item in telemetry if item.get("status") == "complete"]
    judgments = [
        read_json(path)
        for path in sorted((run_dir / "checkpoints" / "judge").glob("*.json"))
    ]
    judgments = [item for item in judgments if item.get("status") == "complete"]
    summary = build_telemetry_summary(records, telemetry, judgments)
    summary_path = run_dir / "telemetry_summary.json"
    report_path = run_dir / "memory_filtering_telemetry_report.md"
    atomic_write_json(summary_path, summary)
    report_path.write_text(
        render_telemetry_report(summary, scenarios, records, telemetry, judgments),
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = load_run_manifest(run_dir)
    manifest["telemetry"] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "status": summary["status"],
        "completed_at": datetime.now(UTC).isoformat(),
        "model": config.judge.model,
    }
    atomic_write_json(manifest_path, manifest)
    store = _store(config, run_dir, upload)
    if store:
        for path in (summary_path, report_path, manifest_path):
            store.upload_file(path, path.name)
    if summary["status"] != "complete" and not allow_partial:
        raise RuntimeError("Telemetry report is incomplete")
    return summary
