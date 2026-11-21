from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


TAXONOMY_GROUPS: dict[str, tuple[str, ...]] = {
    "living_logistics": (
        "residence_housing",
        "home_environment_upkeep",
        "possessions_acquisitions",
        "transportation_commuting",
        "travel_trips",
    ),
    "work_finance": (
        "employment_career",
        "work_tasks_conditions",
        "learning_skills",
        "financial_management",
        "goals_commitments",
    ),
    "health_personal": (
        "physical_health_care",
        "sleep_energy",
        "exercise_activity",
        "food_meals_substances",
        "emotional_wellbeing",
    ),
    "social_other": (
        "relationships_caregiving",
        "social_participation",
        "routines_time",
        "technology_digital_behavior",
        "interests_tastes_values",
    ),
}

GROUP_DISPLAY_NAMES: dict[str, str] = {
    "living_logistics": "Living & logistics",
    "work_finance": "Work & finance",
    "health_personal": "Health & personal",
    "social_other": "Social & other",
}

LABEL_DISPLAY_NAMES: dict[str, str] = {
    "residence_housing": "Residence & housing",
    "home_environment_upkeep": "Home environment & upkeep",
    "possessions_acquisitions": "Possessions & acquisitions",
    "transportation_commuting": "Transportation & commuting",
    "travel_trips": "Travel & trips",
    "employment_career": "Employment & career",
    "work_tasks_conditions": "Work tasks & conditions",
    "learning_skills": "Learning & skills",
    "financial_management": "Financial management",
    "goals_commitments": "Goals & commitments",
    "physical_health_care": "Physical health & care",
    "sleep_energy": "Sleep & energy",
    "exercise_activity": "Exercise & activity",
    "food_meals_substances": "Food, meals & substances",
    "emotional_wellbeing": "Emotional wellbeing",
    "relationships_caregiving": "Relationships & caregiving",
    "social_participation": "Social participation",
    "routines_time": "Routines & time",
    "technology_digital_behavior": "Technology & digital behavior",
    "interests_tastes_values": "Interests, tastes & values",
}

LABEL_TO_GROUP = {
    label: group for group, labels in TAXONOMY_GROUPS.items() for label in labels
}
VALID_LABELS = frozenset(LABEL_TO_GROUP)
_LABEL_ALIASES = {
    re.sub(r"[^a-z0-9]", "", value.lower()): label
    for label, display in LABEL_DISPLAY_NAMES.items()
    for value in (label, display)
}


class TaxonomyValidationError(ValueError):
    """Raised when model-produced memory labels violate the fixed taxonomy."""


def normalize_label_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    key = re.sub(r"[^a-z0-9]", "", value.lower())
    return _LABEL_ALIASES.get(key)


def validate_category_labels(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) != 2:
        raise TaxonomyValidationError("labels must contain exactly two label IDs")
    if not all(isinstance(label, str) for label in value):
        raise TaxonomyValidationError("labels must be strings")
    labels = [normalize_label_id(label) or label.strip() for label in value]
    if len(set(labels)) != 2:
        raise TaxonomyValidationError("labels must be distinct")
    unknown = sorted(set(labels) - VALID_LABELS)
    if unknown:
        parent_buckets = sorted(set(unknown).intersection(TAXONOMY_GROUPS))
        detail = (
            f" Parent buckets are not selectable labels: {parent_buckets}."
            if parent_buckets
            else ""
        )
        raise TaxonomyValidationError(
            f"unknown taxonomy labels: {unknown}.{detail} "
            f"Valid leaf label IDs are: {sorted(VALID_LABELS)}"
        )
    groups = {LABEL_TO_GROUP[label] for label in labels}
    if len(groups) != 2:
        raise TaxonomyValidationError(
            "the two labels must belong to different parent groups"
        )
    return labels


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def repair_category_labels(
    value: Any,
    fact: str,
    embedder: Any,
) -> tuple[list[str], str, str | None]:
    """Return valid labels, repair status, and the original validation error."""
    original = list(value) if isinstance(value, list) else value
    try:
        labels = validate_category_labels(value)
        status = "model" if labels == original else "normalized"
        return labels, status, None
    except TaxonomyValidationError as exc:
        validation_error = str(exc)

    cache = getattr(embedder, "_lightmem_taxonomy_embeddings", None)
    if cache is None:
        cache = {
            label: embedder.embed(
                f"User memory category: {LABEL_DISPLAY_NAMES[label]}"
            )
            for label in sorted(VALID_LABELS)
        }
        setattr(embedder, "_lightmem_taxonomy_embeddings", cache)
    fact_vector = embedder.embed(fact)
    scores = {
        label: _cosine(fact_vector, vector) for label, vector in cache.items()
    }
    normalized_existing = []
    if isinstance(value, list):
        for item in value:
            normalized = normalize_label_id(item)
            if normalized and normalized not in normalized_existing:
                normalized_existing.append(normalized)
    if normalized_existing:
        primary = max(normalized_existing, key=lambda label: scores[label])
    else:
        primary = max(sorted(VALID_LABELS), key=lambda label: scores[label])
    cross_group = [
        label
        for label in VALID_LABELS
        if LABEL_TO_GROUP[label] != LABEL_TO_GROUP[primary]
    ]
    secondary = max(sorted(cross_group), key=lambda label: scores[label])
    labels = [primary, secondary]
    validate_category_labels(labels)
    return labels, "semantic_fallback", validation_error


def repair_extracted_result_labels(
    extracted_results: list[dict[str, Any] | None],
    embedder: Any,
    logger: Any = None,
) -> None:
    for result in extracted_results:
        if not result:
            continue
        for entry in result.get("cleaned_result", []):
            original = entry.get("labels")
            existing_status = entry.get("taxonomy_label_status")
            fact = entry.get("fact") or entry.get("relation") or ""
            labels, status, error = repair_category_labels(
                original, fact, embedder
            )
            entry["labels"] = labels
            if existing_status == "normalized" and status == "model":
                status = existing_status
            entry["taxonomy_label_status"] = status
            if status != "model":
                entry["taxonomy_original_labels"] = original
            if error:
                entry["taxonomy_label_error"] = error
                if logger:
                    logger.warning(
                        "Repaired invalid taxonomy labels %r for fact %r: %s",
                        original,
                        fact[:120],
                        error,
                    )


def validate_extracted_labels(entries: list[dict[str, Any]]) -> None:
    for index, entry in enumerate(entries):
        original = entry.get("labels")
        try:
            entry["labels"] = validate_category_labels(original)
            entry["taxonomy_label_status"] = (
                "model" if entry["labels"] == original else "normalized"
            )
            if entry["taxonomy_label_status"] == "normalized":
                entry["taxonomy_original_labels"] = original
        except TaxonomyValidationError as exc:
            raise TaxonomyValidationError(f"fact {index}: {exc}") from exc


def parse_labeled_extraction_response(response: str) -> list[dict[str, Any]]:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response.strip())
    text = match.group(1).strip() if match else response.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TaxonomyValidationError(f"invalid extraction JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("data"), list):
        raise TaxonomyValidationError(
            "extraction response must be an object containing a data array"
        )
    if not all(isinstance(item, dict) for item in parsed["data"]):
        raise TaxonomyValidationError("every extraction data item must be an object")
    entries = parsed["data"]
    validate_extracted_labels(entries)
    return entries


def label_metadata(labels: list[str]) -> list[dict[str, str]]:
    if not isinstance(labels, list) or not labels:
        raise TaxonomyValidationError("at least one label is required")
    unknown = sorted(set(labels) - VALID_LABELS)
    if unknown:
        raise TaxonomyValidationError(f"unknown taxonomy labels: {unknown}")
    return [
        {
            "id": label,
            "name": LABEL_DISPLAY_NAMES[label],
            "group": LABEL_TO_GROUP[label],
        }
        for label in labels
    ]


def _taxonomy_catalog() -> str:
    lines = [
        "Only the indented leaf IDs below are selectable labels.",
        "The parent bucket headings are organizational text only and MUST NEVER appear in `labels`.",
    ]
    for group, labels in TAXONOMY_GROUPS.items():
        rendered = ", ".join(
            f"{label} ({LABEL_DISPLAY_NAMES[label]})" for label in labels
        )
        lines.append(
            f"- Parent bucket: {GROUP_DISPLAY_NAMES[group]} [NOT A LABEL]\n"
            f"  Selectable leaf IDs: {rendered}"
        )
    return "\n".join(lines)


def taxonomy_prompt() -> str:
    return "\n".join(
        [
            "For every extracted fact, also assign exactly two taxonomy label IDs.",
            "The two labels MUST be distinct and MUST come from different parent groups.",
            "Choose the best label in the primary group and the best cross-group label, even when one group is dominant.",
            "Add them as `labels`: [\"label_id_1\", \"label_id_2\"] on every data item.",
            "Valid example: `labels`: [\"residence_housing\", \"goals_commitments\"].",
            "Parent bucket headings are invalid label values.",
            _taxonomy_catalog(),
        ]
    )


def update_taxonomy_prompt() -> str:
    return "\n".join(
        [
            "",
            "If action is `update`, also return `labels` containing exactly two valid taxonomy label IDs for `new_memory`.",
            "The two labels MUST be distinct and MUST come from different parent groups.",
            "Choose the best label in the primary group and the best cross-group label, even when one group is dominant.",
            "Never return a parent bucket heading as a label value.",
            _taxonomy_catalog(),
        ]
    )


def taxonomy_digest() -> str:
    encoded = json.dumps(
        TAXONOMY_GROUPS, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
