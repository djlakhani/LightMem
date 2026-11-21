from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Callable

from lightmem.memory.taxonomy import label_metadata, validate_category_labels


RELATIONS = frozenset({"none", "updates", "invalidates", "appends"})


def parse_json_object(content: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    text = match.group(1) if match else content
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Response JSON must be an object")
    return value


def build_anchor_selection_messages(
    query: str,
    retrieval: list[dict[str, Any]],
    dimension: str,
    anchor_count: int,
) -> list[dict[str, str]]:
    evidence = [
        {
            "id": str(item.get("id")),
            "timestamp": item.get("payload", {}).get("time_stamp", ""),
            "memory": item.get("payload", {}).get("memory", ""),
        }
        for item in retrieval
    ]
    system = (
        "Select the memories that most directly support answering the query. "
        f"Return only JSON with `anchor_ids`, containing exactly {anchor_count} "
        "unique IDs copied from the supplied memories, ordered most to least useful. "
        "Do not answer the query and do not invent IDs."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {"dimension": dimension, "query": query, "memories": evidence},
                ensure_ascii=False,
            ),
        },
    ]


def parse_anchor_selection(
    content: str, allowed_ids: set[str], anchor_count: int
) -> dict[str, list[str]]:
    value = parse_json_object(content)
    anchor_ids = value.get("anchor_ids")
    if not isinstance(anchor_ids, list) or not all(
        isinstance(item, str) for item in anchor_ids
    ):
        raise ValueError("anchor_ids must be an array of strings")
    if len(anchor_ids) != anchor_count:
        raise ValueError(f"anchor_ids must contain exactly {anchor_count} IDs")
    if len(set(anchor_ids)) != anchor_count:
        raise ValueError("anchor_ids must be unique")
    invented = sorted(set(anchor_ids) - allowed_ids)
    if invented:
        raise ValueError(f"anchor_ids contained IDs not in retrieval: {invented}")
    return {"anchor_ids": anchor_ids}


def _labeled_memory(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload", {})
    labels = validate_category_labels(payload.get("category_labels"))
    return {
        "id": str(item.get("id")),
        "timestamp": payload.get("time_stamp", ""),
        "memory": payload.get("memory", ""),
        "labels": label_metadata(labels),
    }


def build_resolution_messages(
    query: str,
    anchors: list[dict[str, Any]],
    candidate_groups: dict[str, list[dict[str, Any]]],
    max_dependencies: int,
) -> list[dict[str, str]]:
    groups: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_id = str(anchor["id"])
        candidates = []
        for candidate in candidate_groups.get(anchor_id, []):
            rendered = _labeled_memory(candidate)
            rendered["matching_labels"] = candidate["matching_labels"]
            rendered["selection_source"] = candidate["selection_source"]
            candidates.append(rendered)
        groups.append(
            {
                "anchor": _labeled_memory(anchor),
                "future_states_within_matching_categories": candidates,
            }
        )

    system = f"""You resolve how later user states affect evidence selected for a query.
Each candidate is later than its anchor and was retrieved because it shares at least one category label. Evaluate every supplied anchor-candidate pair; category overlap is only a reason to inspect a pair, not proof of a relationship.

For every candidate, assign exactly one relation:
- `updates`: changes or refines the same user state.
- `invalidates`: makes the anchor obsolete or false.
- `appends`: adds material context that should be combined with the anchor.
- `none`: does not materially alter or extend the anchor.

Return only JSON with `anchor_resolutions`. For each anchor return:
- `anchor_id`
- `candidate_verdicts`: every supplied candidate exactly once, each with `memory_id` and `relation`
- `selected_changes`: at most {max_dependencies} strongest non-none candidates, each with `memory_id`, `relation`, and a specific `why`
- `resolution_summary`: a substantive statement of the resulting current state. If all candidates are `none`, explain why the later same-category states do not change the anchor.

Do not use `none` merely because comparison is difficult. Do not invent facts or IDs."""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {"query": query, "anchor_groups": groups}, ensure_ascii=False
            ),
        },
    ]


def parse_resolution_response(
    content: str,
    anchors: list[dict[str, Any]],
    candidate_groups: dict[str, list[dict[str, Any]]],
    max_dependencies: int,
) -> dict[str, Any]:
    value = parse_json_object(content)
    resolutions = value.get("anchor_resolutions")
    if not isinstance(resolutions, list):
        raise ValueError("anchor_resolutions must be an array")
    expected_anchor_ids = [str(anchor["id"]) for anchor in anchors]
    by_anchor: dict[str, dict[str, Any]] = {}
    for resolution in resolutions:
        if not isinstance(resolution, dict):
            raise ValueError("each anchor resolution must be an object")
        anchor_id = resolution.get("anchor_id")
        if anchor_id in by_anchor:
            raise ValueError(f"duplicate anchor resolution: {anchor_id}")
        if anchor_id not in expected_anchor_ids:
            raise ValueError(f"unknown anchor resolution: {anchor_id}")
        expected_candidates = {
            str(item["id"]) for item in candidate_groups.get(anchor_id, [])
        }
        verdicts = resolution.get("candidate_verdicts")
        if not isinstance(verdicts, list):
            raise ValueError(f"candidate_verdicts must be an array for {anchor_id}")
        verdict_map: dict[str, str] = {}
        for verdict in verdicts:
            if not isinstance(verdict, dict):
                raise ValueError("candidate verdicts must be objects")
            memory_id = verdict.get("memory_id")
            relation = verdict.get("relation")
            if memory_id in verdict_map:
                raise ValueError(f"duplicate candidate verdict: {memory_id}")
            if memory_id not in expected_candidates:
                raise ValueError(f"candidate {memory_id} does not belong to {anchor_id}")
            if relation not in RELATIONS:
                raise ValueError(f"invalid relation for {memory_id}: {relation}")
            verdict_map[memory_id] = relation
        if set(verdict_map) != expected_candidates:
            missing = sorted(expected_candidates - set(verdict_map))
            raise ValueError(f"resolver skipped candidates for {anchor_id}: {missing}")

        selected = resolution.get("selected_changes")
        if not isinstance(selected, list):
            raise ValueError(f"selected_changes must be an array for {anchor_id}")
        if len(selected) > max_dependencies:
            raise ValueError(f"too many selected changes for {anchor_id}")
        selected_ids: set[str] = set()
        normalized_selected: list[dict[str, str]] = []
        for change in selected:
            if not isinstance(change, dict):
                raise ValueError("selected changes must be objects")
            memory_id = change.get("memory_id")
            relation = change.get("relation")
            why = change.get("why")
            if memory_id in selected_ids:
                raise ValueError(f"duplicate selected change: {memory_id}")
            if verdict_map.get(memory_id) != relation or relation == "none":
                raise ValueError(
                    f"selected relation for {memory_id} disagrees with its verdict"
                )
            if not isinstance(why, str) or len(why.strip()) < 20:
                raise ValueError(f"selected change {memory_id} needs a specific why")
            selected_ids.add(memory_id)
            normalized_selected.append(
                {"memory_id": memory_id, "relation": relation, "why": why.strip()}
            )
        has_non_none = any(relation != "none" for relation in verdict_map.values())
        if has_non_none and not normalized_selected:
            raise ValueError(f"{anchor_id} has non-none verdicts but no selected changes")
        if not has_non_none and normalized_selected:
            raise ValueError(f"{anchor_id} selected a change despite all-none verdicts")

        summary = resolution.get("resolution_summary")
        if not isinstance(summary, str) or len(summary.strip()) < 30:
            raise ValueError(f"{anchor_id} needs a substantive resolution_summary")
        by_anchor[anchor_id] = {
            "anchor_id": anchor_id,
            "candidate_verdicts": [
                {"memory_id": memory_id, "relation": verdict_map[memory_id]}
                for memory_id in [
                    str(item["id"])
                    for item in candidate_groups.get(anchor_id, [])
                ]
            ],
            "selected_changes": normalized_selected,
            "resolution_summary": summary.strip(),
        }
    if set(by_anchor) != set(expected_anchor_ids):
        missing = sorted(set(expected_anchor_ids) - set(by_anchor))
        raise ValueError(f"resolver omitted anchors: {missing}")
    return {"anchor_resolutions": [by_anchor[item] for item in expected_anchor_ids]}


def build_final_answer_messages(
    query: str,
    dimension: str,
    anchors: list[dict[str, Any]],
    candidate_groups: dict[str, list[dict[str, Any]]],
    resolution: dict[str, Any],
) -> list[dict[str, str]]:
    candidate_maps = {
        anchor_id: {str(item["id"]): item for item in candidates}
        for anchor_id, candidates in candidate_groups.items()
    }
    packages: list[dict[str, Any]] = []
    for anchor, resolved in zip(anchors, resolution["anchor_resolutions"]):
        anchor_id = str(anchor["id"])
        future_states = []
        for change in resolved["selected_changes"]:
            candidate = candidate_maps[anchor_id][change["memory_id"]]
            rendered = _labeled_memory(candidate)
            rendered.update(
                {
                    "matching_labels": candidate["matching_labels"],
                    "relation": change["relation"],
                    "why": change["why"],
                }
            )
            future_states.append(rendered)
        packages.append(
            {
                "anchor": _labeled_memory(anchor),
                "selected_later_states": future_states,
                "category_resolution": resolved["resolution_summary"],
            }
        )

    if dimension == "IPA":
        task = "respond directly to the user's latest query"
        query_label = "latest_query"
    else:
        task = "accurately answer the question"
        query_label = "question"
    system = (
        f"Use the resolved memory evidence to {task}. Later states marked `updates` "
        "or `invalidates` take precedence over their anchor; combine `appends`. When "
        "a package has no selected later state, use its anchor consistently with the "
        "category resolution. Return only the answer, without memory IDs, labels, or "
        "an explanation of the memory-resolution process."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {query_label: query, "resolved_evidence": packages},
                ensure_ascii=False,
            ),
        },
    ]


def generate_validated_json(
    qwen: Any,
    messages: list[dict[str, str]],
    parser: Callable[[str], dict[str, Any]],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    active_messages = list(messages)
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
        generation = qwen.generate(
            active_messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        generation_data = asdict(generation)
        try:
            parsed = parser(generation.content)
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
            if attempt == 1:
                raise RuntimeError(
                    f"Structured generation remained invalid: {exc}"
                ) from exc
            active_messages.extend(
                [
                    {"role": "assistant", "content": generation.content},
                    {
                        "role": "user",
                        "content": (
                            f"Your JSON was invalid: {exc}. Return the complete corrected "
                            "JSON object only and obey every ID and cardinality constraint."
                        ),
                    },
                ]
            )
    raise AssertionError("unreachable")
