from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from typing import Any

EXPERIMENT_VERSION = "stale-evidence-attribution-v1"
DIMENSIONS = (
    ("dim1", "SR", "State Recognition"),
    ("dim2", "PR", "Premise Resistance"),
    ("dim3", "IPA", "Implicitly Personalized Action"),
)
MESSAGE_LINE = re.compile(
    r"^\[(?P<timestamp>[^,\]]+),\s*(?P<weekday>[^\]]+)\]\s+"
    r"(?P<source_id>\d+)\.(?P<speaker>[^:]+):\s*(?P<content>.*)$"
)
WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def _normalized(text: str) -> str:
    return " ".join(WORD.findall(text.lower().replace("’", "'")))


def _similarity(left: str, right: str) -> float:
    a = _normalized(left)
    b = _normalized(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    sequence = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    overlap = len(a_tokens & b_tokens)
    containment = overlap / min(len(a_tokens), len(b_tokens))
    jaccard = overlap / len(a_tokens | b_tokens)
    return max(sequence, 0.75 * containment + 0.25 * jaccard)


def _minute(timestamp: str) -> str:
    return timestamp.replace(" ", "T")[:16]


def _parse_extraction_messages(prompt: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in prompt.splitlines():
        match = MESSAGE_LINE.match(line)
        if match:
            current = {
                "timestamp": match.group("timestamp"),
                "weekday": match.group("weekday"),
                "source_id": int(match.group("source_id")),
                "speaker": match.group("speaker"),
                "content": match.group("content"),
            }
            parsed.append(current)
        elif current is not None and line and not line.startswith("--- Topic"):
            current["content"] += "\n" + line
    return parsed


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


def _parse_extraction_output(raw: str) -> list[dict[str, Any]]:
    try:
        payload = parse_json_object(raw)
    except (ValueError, json.JSONDecodeError):
        return []
    data = payload.get("data", [])
    return [item for item in data if isinstance(item, dict)]


def _extraction_batches(record: dict[str, Any]) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    calls = record.get("ingest", {}).get("extraction_calls", [])
    for call_index, call in enumerate(calls, start=1):
        inputs = call.get("add_input_prompt", [])
        outputs = call.get("add_output_prompt", [])
        for batch_index, input_messages in enumerate(inputs, start=1):
            if not isinstance(input_messages, list):
                continue
            user_prompt = next(
                (
                    item.get("content", "")
                    for item in input_messages
                    if isinstance(item, dict) and item.get("role") == "user"
                ),
                "",
            )
            raw_output = outputs[batch_index - 1] if batch_index <= len(outputs) else ""
            batches.append(
                {
                    "call_index": call_index,
                    "batch_index": batch_index,
                    "messages": _parse_extraction_messages(user_prompt),
                    "facts": (
                        _parse_extraction_output(raw_output)
                        if isinstance(raw_output, str)
                        else []
                    ),
                }
            )
    return batches


def _source_messages(scenario: dict[str, Any], view: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for session_index, session in enumerate(scenario["views"][view]["sessions"]):
        for message_index, message in enumerate(session["messages"]):
            if message.get("role") != "user":
                continue
            result.append(
                {
                    "session_index": session_index,
                    "message_index": message_index,
                    "timestamp": session["timestamp"],
                    "content": message.get("content", ""),
                }
            )
    return result


def discover_state_candidates(
    target: str,
    scenario: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Trace a ground-truth state to source-attributed stored memory chunks."""
    view = record.get("context_view", "truncated")
    originals = _source_messages(scenario, view)
    if not originals:
        return {
            "target": target,
            "status": "no_source_message",
            "candidates": [],
            "eligible_candidates": [],
        }
    source_score, source = max(
        ((_similarity(target, item["content"]), item) for item in originals),
        key=lambda item: item[0],
    )
    prompt_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for batch in _extraction_batches(record):
        for message in batch["messages"]:
            if _minute(message["timestamp"]) != _minute(source["timestamp"]):
                continue
            score = max(
                _similarity(source["content"], message["content"]),
                _similarity(target, message["content"]),
            )
            prompt_candidates.append((score, batch, message))
    if not prompt_candidates:
        return {
            "target": target,
            "status": "missing_from_extraction_input",
            "source": source,
            "source_score": source_score,
            "candidates": [],
            "eligible_candidates": [],
        }
    prompt_score, batch, prompt_message = max(
        prompt_candidates, key=lambda item: item[0]
    )
    if prompt_score < 0.48:
        return {
            "target": target,
            "status": "missing_from_extraction_input",
            "source": source,
            "source_score": source_score,
            "closest_extraction_message": prompt_message,
            "extraction_score": prompt_score,
            "candidates": [],
            "eligible_candidates": [],
        }

    facts = [
        item.get("fact") or item.get("relation")
        for item in batch["facts"]
        if int(item.get("source_id", -1)) == prompt_message["source_id"]
        and (item.get("fact") or item.get("relation"))
    ]
    pre_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for memory in record.get("pre_update_memories", []):
        pre_by_text[memory.get("payload", {}).get("memory", "")].append(memory)
    post_by_id = {
        str(memory["id"]): memory for memory in record.get("post_update_memories", [])
    }
    candidates: list[dict[str, Any]] = []
    unmatched_facts: list[str] = []
    for fact in facts:
        matches = pre_by_text.get(fact, [])
        if not matches:
            unmatched_facts.append(fact)
        for before in matches:
            memory_id = str(before["id"])
            after = post_by_id.get(memory_id)
            before_text = before.get("payload", {}).get("memory", "")
            after_text = (
                after.get("payload", {}).get("memory", "")
                if after is not None
                else None
            )
            if after is None:
                consolidation_status = "deleted"
            elif after_text != before_text:
                consolidation_status = "updated"
            else:
                consolidation_status = "unchanged"
            candidates.append(
                {
                    "id": memory_id,
                    "extracted_fact": fact,
                    "pre_memory": before_text,
                    "post_memory": after_text,
                    "timestamp": before.get("payload", {}).get("time_stamp"),
                    "consolidation_status": consolidation_status,
                    "retrievable": after is not None,
                }
            )
    eligible = [item for item in candidates if item["retrievable"]]
    return {
        "target": target,
        "status": "candidates_found" if candidates else "no_persisted_facts",
        "source": source,
        "source_score": source_score,
        "extraction_message": prompt_message,
        "extraction_score": prompt_score,
        "extraction_call": batch["call_index"],
        "extraction_batch": batch["batch_index"],
        "facts": facts,
        "unmatched_facts": unmatched_facts,
        "candidates": candidates,
        "eligible_candidates": eligible,
    }


def build_citation_prompt(
    query: str,
    retrieval: list[dict[str, Any]],
    dimension: str,
    citation_count: int = 4,
) -> list[dict[str, str]]:
    if dimension == "IPA":
        task = "respond to the user's latest query directly"
        query_label = "Latest Query"
    else:
        task = "accurately answer the question"
        query_label = "Question"
    system = (
        f"You are a helpful assistant. Review the retrieved memories and {task}. "
        f"Also identify exactly {citation_count} memory IDs from the supplied evidence "
        "that most influenced your judgment, ordered from most to least important. "
        "Use only supplied IDs. Return only one JSON object with keys `answer` "
        "(a string) and `memory_ids_used` (an array of unique strings)."
    )
    evidence: list[str] = []
    for item in retrieval:
        payload = item.get("payload", {})
        evidence.extend(
            [
                f"[MEMORY_ID={item.get('id')}]",
                (
                    f"Timestamp: {payload.get('time_stamp', '')} "
                    f"{payload.get('weekday', '')}"
                ).strip(),
                f"Memory: {payload.get('memory', '')}",
                "",
            ]
        )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "[Retrieved Evidence]\n"
                + "\n".join(evidence).rstrip()
                + f"\n\n[{query_label}]\n{query}"
            ),
        },
    ]


def parse_citation_response(
    content: str,
    allowed_ids: set[str],
    citation_count: int = 4,
) -> dict[str, Any]:
    value = parse_json_object(content)
    answer = value.get("answer")
    ids = value.get("memory_ids_used")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Citation response must contain a nonempty answer string")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("memory_ids_used must be an array of strings")
    if len(ids) != citation_count:
        raise ValueError(f"memory_ids_used must contain exactly {citation_count} IDs")
    if len(set(ids)) != citation_count:
        raise ValueError("memory_ids_used must contain unique IDs")
    invented = sorted(set(ids) - allowed_ids)
    if invented:
        raise ValueError(f"memory_ids_used contained IDs not in retrieval: {invented}")
    return {"answer": answer.strip(), "memory_ids_used": ids}


def citation_metrics(
    selected_ids: list[str],
    retrieval: list[dict[str, Any]],
    old_ids: set[str],
    new_ids: set[str],
) -> dict[str, Any]:
    retrieved_ids = [str(item.get("id")) for item in retrieval]
    retrieved_set = set(retrieved_ids)
    old_available = [item for item in retrieved_ids if item in old_ids]
    new_available = [item for item in retrieved_ids if item in new_ids]
    selected_old = [item for item in selected_ids if item in old_ids]
    selected_new = [item for item in selected_ids if item in new_ids]
    selected_neither = [
        item for item in selected_ids if item not in old_ids and item not in new_ids
    ]
    if selected_old and selected_new:
        pattern = "both"
    elif selected_old:
        pattern = "old_only"
    elif selected_new:
        pattern = "new_only"
    else:
        pattern = "neither"
    return {
        "old_ground_truth_ids": sorted(old_ids),
        "new_ground_truth_ids": sorted(new_ids),
        "old_ids_available_in_retrieval": old_available,
        "new_ids_available_in_retrieval": new_available,
        "old_ground_truth_available": bool(old_ids & retrieved_set),
        "new_ground_truth_available": bool(new_ids & retrieved_set),
        "selected_old_ids": selected_old,
        "selected_new_ids": selected_new,
        "selected_neither_ids": selected_neither,
        "used_old_evidence": bool(selected_old),
        "used_new_evidence": bool(selected_new),
        "evidence_pattern": pattern,
        "old_available_recall": (
            len(set(selected_old) & set(old_available)) / len(set(old_available))
            if old_available
            else None
        ),
        "new_available_recall": (
            len(set(selected_new) & set(new_available)) / len(set(new_available))
            if new_available
            else None
        ),
    }
