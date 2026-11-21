from __future__ import annotations

import hashlib
import json
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


class TaxonomyValidationError(ValueError):
    """Raised when model-produced memory labels violate the fixed taxonomy."""


def validate_category_labels(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) != 2:
        raise TaxonomyValidationError("labels must contain exactly two label IDs")
    if not all(isinstance(label, str) for label in value):
        raise TaxonomyValidationError("labels must be strings")
    labels = [label.strip() for label in value]
    if len(set(labels)) != 2:
        raise TaxonomyValidationError("labels must be distinct")
    unknown = sorted(set(labels) - VALID_LABELS)
    if unknown:
        raise TaxonomyValidationError(f"unknown taxonomy labels: {unknown}")
    groups = {LABEL_TO_GROUP[label] for label in labels}
    if len(groups) != 2:
        raise TaxonomyValidationError(
            "the two labels must belong to different parent groups"
        )
    return labels


def validate_extracted_labels(entries: list[dict[str, Any]]) -> None:
    for index, entry in enumerate(entries):
        try:
            entry["labels"] = validate_category_labels(entry.get("labels"))
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
    lines = ["Only use the following label IDs:"]
    for group, labels in TAXONOMY_GROUPS.items():
        rendered = ", ".join(
            f"{label} ({LABEL_DISPLAY_NAMES[label]})" for label in labels
        )
        lines.append(f"- {group}: {rendered}")
    return "\n".join(lines)


def taxonomy_prompt() -> str:
    return "\n".join(
        [
            "For every extracted fact, also assign exactly two taxonomy label IDs.",
            "The two labels MUST be distinct and MUST come from different parent groups.",
            "Choose the best label in the primary group and the best cross-group label, even when one group is dominant.",
            "Add them as `labels`: [\"label_id_1\", \"label_id_2\"] on every data item.",
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
            _taxonomy_catalog(),
        ]
    )


def taxonomy_digest() -> str:
    encoded = json.dumps(
        TAXONOMY_GROUPS, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
