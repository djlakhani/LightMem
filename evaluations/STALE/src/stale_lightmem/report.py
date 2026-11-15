from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json

DIMENSION_NAMES = {"dim1": "SR", "dim2": "PR", "dim3": "IPA"}


def _bucket() -> dict[str, int]:
    return {"correct": 0, "total": 0}


def build_summary(
    judgments: Iterable[dict[str, Any]],
    scenario_records: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = defaultdict(_bucket)
    judgment_list = list(judgments)
    for record in judgment_list:
        scenario_type = record["type"]
        evaluation = record["evaluation"]
        for dim_key, dim_name in DIMENSION_NAMES.items():
            passed = evaluation[f"{dim_key}_eval"]["pass"]
            for key in (
                "overall",
                scenario_type,
                dim_name,
                f"{scenario_type}×{dim_name}",
            ):
                buckets[key]["total"] += 1
                buckets[key]["correct"] += int(passed)

    ordered_keys = [
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
    for key in ordered_keys:
        value = buckets[key]
        accuracy[key] = {
            **value,
            "accuracy": value["correct"] / value["total"] if value["total"] else None,
        }

    records = list(scenario_records)
    memory_totals = {
        "before": sum(
            item.get("memory_diff", {}).get("before_count", 0) for item in records
        ),
        "after": sum(
            item.get("memory_diff", {}).get("after_count", 0) for item in records
        ),
        "deleted": sum(
            len(item.get("memory_diff", {}).get("deleted_ids", [])) for item in records
        ),
        "updated": sum(
            len(item.get("memory_diff", {}).get("updated_ids", [])) for item in records
        ),
    }
    timing = {
        "scenario_total_seconds": sum(
            item.get("elapsed_seconds", 0.0) for item in records
        ),
        "ingest_seconds": sum(
            item.get("ingest", {}).get("elapsed_seconds", 0.0) for item in records
        ),
        "consolidation_seconds": sum(
            item.get("consolidation", {}).get("elapsed_seconds", 0.0)
            for item in records
        ),
        "retrieval_seconds": sum(
            answer.get("retrieval_elapsed_seconds", 0.0)
            for item in records
            for answer in item.get("answers", {}).values()
        ),
        "answer_generation_seconds": sum(
            answer.get("generation", {}).get("elapsed_seconds", 0.0)
            for item in records
            for answer in item.get("answers", {}).values()
        ),
        "judge_seconds": sum(
            item.get("elapsed_seconds", 0.0) for item in judgment_list
        ),
    }
    usage = {
        "answer_qwen_tokens": sum(
            answer.get("generation", {}).get("usage", {}).get("total_tokens", 0)
            for item in records
            for answer in item.get("answers", {}).values()
        ),
        "memory_qwen_tokens": sum(
            item.get("token_statistics", {})
            .get("summary", {})
            .get("total_llm_tokens", 0)
            for item in records
        ),
        "embedding_calls": sum(
            item.get("token_statistics", {})
            .get("summary", {})
            .get("total_embedding_calls", 0)
            for item in records
        ),
        "judge_tokens": sum(
            item.get("usage", {}).get("total_tokens", 0) for item in judgment_list
        ),
    }
    return {
        "status": "complete" if len(judgment_list) == 50 else "partial",
        "judged_scenarios": len(judgment_list),
        "judged_queries": len(judgment_list) * 3,
        "accuracy": accuracy,
        "memory": memory_totals,
        "timing": timing,
        "usage": usage,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LightMem STALE Evaluation",
        "",
        f"Status: **{summary['status']}**",
        "",
        "## Accuracy",
        "",
        "| Section | Passed | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for key, value in summary["accuracy"].items():
        accuracy = (
            "n/a" if value["accuracy"] is None else f"{100 * value['accuracy']:.2f}%"
        )
        lines.append(f"| {key} | {value['correct']} | {value['total']} | {accuracy} |")
    memory = summary["memory"]
    lines.extend(
        [
            "",
            "## Memory consolidation",
            "",
            f"- Before update: {memory['before']}",
            f"- After update: {memory['after']}",
            f"- Deleted: {memory['deleted']}",
            f"- Updated: {memory['updated']}",
            "",
            "## Efficiency",
            "",
            f"- Memory construction: {summary['timing']['ingest_seconds']:.2f}s",
            f"- Offline consolidation: {summary['timing']['consolidation_seconds']:.2f}s",
            f"- Retrieval: {summary['timing']['retrieval_seconds']:.2f}s",
            f"- Answer generation: {summary['timing']['answer_generation_seconds']:.2f}s",
            f"- Judge: {summary['timing']['judge_seconds']:.2f}s",
            f"- Memory-system Qwen tokens: {summary['usage']['memory_qwen_tokens']}",
            f"- Answer Qwen tokens: {summary['usage']['answer_qwen_tokens']}",
            f"- Judge tokens: {summary['usage']['judge_tokens']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(run_dir: Path, summary: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
