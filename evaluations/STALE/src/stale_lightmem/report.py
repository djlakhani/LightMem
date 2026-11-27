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
    selection_results = [
        answer.get("evidence_selection", {}).get("generation", {})
        for item in records
        for answer in item.get("answers", {}).values()
        if answer.get("evidence_selection")
    ]
    evidence_selection = {
        "total": len(selection_results),
        "llm": sum(result.get("method") == "llm" for result in selection_results),
        "embedding_rank_fallback": sum(
            bool(result.get("fallback_used")) for result in selection_results
        ),
    }
    future_anchors = [
        anchor
        for item in records
        for answer in item.get("answers", {}).values()
        for anchor in answer.get("future_update_analysis", {}).get("anchors", [])
    ]
    future_updates = {
        "anchors": len(future_anchors),
        "with_candidates": sum(
            bool(anchor.get("candidate_search", {}).get("candidates"))
            for anchor in future_anchors
        ),
        "selected_updates": sum(
            len(anchor.get("selected_updates", [])) for anchor in future_anchors
        ),
        "valid_empty": sum(
            anchor.get("filter_result", {}).get("method") == "llm"
            and not anchor.get("selected_updates")
            for anchor in future_anchors
        ),
        "no_candidates": sum(
            anchor.get("filter_result", {}).get("method") == "no_candidates"
            for anchor in future_anchors
        ),
        "repair_successes": sum(
            anchor.get("filter_result", {}).get("method") == "llm"
            and anchor.get("filter_result", {}).get("llm_attempts") == 2
            for anchor in future_anchors
        ),
        "no_updates_fallback": sum(
            bool(anchor.get("filter_result", {}).get("fallback_used"))
            for anchor in future_anchors
        ),
    }
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
        "evidence_selection_seconds": sum(
            answer.get("evidence_selection", {})
            .get("generation", {})
            .get("elapsed_seconds", 0.0)
            for item in records
            for answer in item.get("answers", {}).values()
        ),
        "future_candidate_search_seconds": sum(
            answer.get("future_update_analysis", {}).get(
                "candidate_search_elapsed_seconds", 0.0
            )
            for item in records
            for answer in item.get("answers", {}).values()
        ),
        "future_filter_seconds": sum(
            anchor.get("filter_result", {}).get("elapsed_seconds", 0.0)
            for anchor in future_anchors
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
        "evidence_selection_qwen_tokens": sum(
            answer.get("evidence_selection", {})
            .get("generation", {})
            .get("usage", {})
            .get("total_tokens", 0)
            for item in records
            for answer in item.get("answers", {}).values()
        ),
        "future_filter_qwen_tokens": sum(
            anchor.get("filter_result", {}).get("usage", {}).get("total_tokens", 0)
            for anchor in future_anchors
        ),
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
        "evidence_selection": evidence_selection,
        "future_updates": future_updates,
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
    evidence_selection = summary["evidence_selection"]
    future_updates = summary["future_updates"]
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
            "## Evidence selection",
            "",
            f"- Questions: {evidence_selection['total']}",
            f"- LLM selections: {evidence_selection['llm']}",
            f"- Embedding-rank fallbacks: {evidence_selection['embedding_rank_fallback']}",
            "",
            "## Future-state reconciliation",
            "",
            f"- Anchors: {future_updates['anchors']}",
            f"- Anchors with candidates: {future_updates['with_candidates']}",
            f"- Selected updates: {future_updates['selected_updates']}",
            f"- Valid empty selections: {future_updates['valid_empty']}",
            f"- No-candidate skips: {future_updates['no_candidates']}",
            f"- Successful repairs: {future_updates['repair_successes']}",
            f"- No-update fallbacks: {future_updates['no_updates_fallback']}",
            "",
            "## Efficiency",
            "",
            f"- Memory construction: {summary['timing']['ingest_seconds']:.2f}s",
            f"- Offline consolidation: {summary['timing']['consolidation_seconds']:.2f}s",
            f"- Retrieval: {summary['timing']['retrieval_seconds']:.2f}s",
            f"- LLM evidence selection: {summary['timing']['evidence_selection_seconds']:.2f}s",
            f"- Future candidate search: {summary['timing']['future_candidate_search_seconds']:.2f}s",
            f"- Future update filtering: {summary['timing']['future_filter_seconds']:.2f}s",
            f"- Answer generation: {summary['timing']['answer_generation_seconds']:.2f}s",
            f"- Judge: {summary['timing']['judge_seconds']:.2f}s",
            f"- Memory-system Qwen tokens: {summary['usage']['memory_qwen_tokens']}",
            f"- Evidence-selection Qwen tokens: {summary['usage']['evidence_selection_qwen_tokens']}",
            f"- Future-filter Qwen tokens: {summary['usage']['future_filter_qwen_tokens']}",
            f"- Answer Qwen tokens: {summary['usage']['answer_qwen_tokens']}",
            f"- Judge tokens: {summary['usage']['judge_tokens']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(run_dir: Path, summary: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
