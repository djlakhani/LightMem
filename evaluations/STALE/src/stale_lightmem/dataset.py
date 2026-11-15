from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DIMENSIONS = (
    ("dim1", "SR", "dim1_query"),
    ("dim2", "PR", "dim2_query"),
    ("dim3", "IPA", "dim3_query"),
)


@dataclass(frozen=True)
class ScenarioDataset:
    metadata: dict[str, Any]
    scenarios: list[dict[str, Any]]
    sha256: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(
    path: str | Path, expected_sha256: str | None = None
) -> ScenarioDataset:
    source = Path(path)
    actual_sha = file_sha256(source)
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(
            f"Scenario SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_dataset(payload)
    metadata = {key: value for key, value in payload.items() if key != "data"}
    return ScenarioDataset(
        metadata=metadata, scenarios=payload["data"], sha256=actual_sha
    )


def validate_dataset(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 2:
        raise ValueError("STALE scenario schema_version must be 2")
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise ValueError("STALE scenario payload must contain a nonempty data list")

    seen: set[str] = set()
    for record in records:
        uid = record.get("uid")
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError(f"Invalid or duplicate scenario uid: {uid!r}")
        seen.add(uid)
        if record.get("type") not in {"T1", "T2"}:
            raise ValueError(f"Scenario {uid} has invalid type")
        for key in ("M_old", "M_new", "explanation"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Scenario {uid} is missing {key}")
        queries = record.get("probing_queries")
        if not isinstance(queries, dict):
            raise ValueError(f"Scenario {uid} is missing probing_queries")
        for _, _, query_key in DIMENSIONS:
            if not isinstance(queries.get(query_key), str) or not queries[query_key]:
                raise ValueError(f"Scenario {uid} is missing {query_key}")
        views = record.get("views", {})
        for view_name in ("truncated", "evidence_only"):
            if view_name not in views:
                raise ValueError(f"Scenario {uid} is missing view {view_name}")
            _validate_sessions(uid, views[view_name].get("sessions"))


def _validate_sessions(uid: str, sessions: Any) -> None:
    if not isinstance(sessions, list) or not sessions:
        raise ValueError(f"Scenario {uid} has no sessions")
    timestamps: list[datetime] = []
    for session in sessions:
        raw_timestamp = session.get("timestamp")
        try:
            timestamps.append(datetime.fromisoformat(raw_timestamp))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Scenario {uid} has invalid timestamp {raw_timestamp!r}"
            ) from exc
        messages = session.get("messages")
        if not isinstance(messages, list) or not messages or len(messages) % 2:
            raise ValueError(f"Scenario {uid} has an empty or odd-length session")
        for index, message in enumerate(messages):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if message.get("role") != expected_role:
                raise ValueError(
                    f"Scenario {uid} message {index} expected role {expected_role}"
                )
            if not isinstance(message.get("content"), str):
                raise ValueError(f"Scenario {uid} message {index} has invalid content")
    if timestamps != sorted(timestamps):
        raise ValueError(f"Scenario {uid} sessions are not chronological")


def iter_turns(
    record: dict[str, Any], view: str
) -> Iterable[tuple[list[dict[str, Any]], bool]]:
    """Yield copied message pairs and whether a pair is the final scenario turn."""
    sessions = record["views"][view]["sessions"]
    total_turns = sum(len(session["messages"]) // 2 for session in sessions)
    emitted = 0
    for session in sessions:
        timestamp = session["timestamp"]
        messages = session["messages"]
        for index in range(0, len(messages), 2):
            pair = copy.deepcopy(messages[index : index + 2])
            for message in pair:
                message["time_stamp"] = timestamp
            emitted += 1
            yield pair, emitted == total_turns
