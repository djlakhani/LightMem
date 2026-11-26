import math
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ATTRIBUTE_TAXONOMY: Mapping[str, Tuple[str, ...]] = {
    "Living & logistics": (
        "Residence & housing",
        "Home environment & upkeep",
        "Possessions & acquisitions",
        "Transportation & commuting",
        "Travel & trips",
    ),
    "Work & finance": (
        "Employment & career",
        "Work tasks & conditions",
        "Learning & skills",
        "Financial management",
        "Goals & commitments",
    ),
    "Health & personal": (
        "Physical health & care",
        "Sleep & energy",
        "Exercise & activity",
        "Food, meals & substances",
        "Emotional wellbeing",
    ),
    "Social & other": (
        "Relationships & caregiving",
        "Social participation",
        "Routines & time",
        "Technology & digital behavior",
        "Interests, tastes & values",
    ),
}

ATTRIBUTE_TO_GROUP: Dict[str, str] = {
    attribute: group
    for group, attributes in ATTRIBUTE_TAXONOMY.items()
    for attribute in attributes
}

ATTRIBUTE_ORDER: Tuple[str, ...] = tuple(ATTRIBUTE_TO_GROUP)

ATTRIBUTE_STATS_KEYS: Tuple[str, ...] = (
    "candidate_facts",
    "llm_valid_pairs",
    "embedding_fallback_attempts",
    "embedding_fallback_successes",
    "embedding_fallback_failures",
    "attribute_rejections",
)


def empty_attribute_statistics() -> Dict[str, int]:
    return {key: 0 for key in ATTRIBUTE_STATS_KEYS}


def validate_attribute_pair(raw_attributes: Any) -> Optional[List[str]]:
    """Return a canonical pair when the LLM output satisfies the taxonomy."""
    if not isinstance(raw_attributes, list) or len(raw_attributes) != 2:
        return None
    if not all(isinstance(attribute, str) for attribute in raw_attributes):
        return None

    attributes = [attribute.strip() for attribute in raw_attributes]
    if any(attribute not in ATTRIBUTE_TO_GROUP for attribute in attributes):
        return None
    if attributes[0] == attributes[1]:
        return None
    if ATTRIBUTE_TO_GROUP[attributes[0]] == ATTRIBUTE_TO_GROUP[attributes[1]]:
        return None
    return attributes


def attribute_prompt_suffix() -> str:
    taxonomy_lines = []
    for group, attributes in ATTRIBUTE_TAXONOMY.items():
        taxonomy_lines.append(f"- {group}: " + "; ".join(attributes))

    return """

ATTRIBUTE LABELING REQUIREMENT (MANDATORY)
For every extracted fact or relation, add an `attributes` field containing exactly
two attribute strings. Order them from most relevant to second-most relevant.
The strings must be copied exactly from the taxonomy below, and the two selected
attributes must belong to different groups. Never invent, paraphrase, or omit an
attribute.

Taxonomy:
{taxonomy}

Required item shapes:
- Factual: {{"source_id": <source_id>, "fact": "<fact>", "attributes": ["<primary>", "<secondary>"]}}
- Relational: {{"source_id": <source_id>, "relation": "<relation>", "attributes": ["<primary>", "<secondary>"]}}
""".format(taxonomy="\n".join(taxonomy_lines))


def add_attribute_instructions(prompt: str) -> str:
    """Append the mandatory taxonomy contract to a default or custom prompt."""
    return f"{prompt.rstrip()}\n{attribute_prompt_suffix()}"


def _as_finite_vector(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Embedding must be a non-empty list or tuple")
    vector = [float(component) for component in value]
    if not all(math.isfinite(component) for component in vector):
        raise ValueError("Embedding contains a non-finite value")
    if not any(component != 0.0 for component in vector):
        raise ValueError("Embedding has zero norm")
    return vector


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Embedding has zero norm")
    score = dot / (left_norm * right_norm)
    if not math.isfinite(score):
        raise ValueError("Cosine similarity is not finite")
    return score


class AttributeLabeler:
    """Validate LLM labels and recover invalid pairs with embedding similarity."""

    def __init__(self, text_embedder: Any):
        if text_embedder is None or not hasattr(text_embedder, "embed"):
            raise ValueError("Attribute labeling requires a configured text embedder")
        self.text_embedder = text_embedder
        self._taxonomy_embeddings: Optional[Dict[str, List[float]]] = None
        self._cache_lock = threading.Lock()

    def _get_taxonomy_embeddings(self) -> Dict[str, List[float]]:
        if self._taxonomy_embeddings is not None:
            return self._taxonomy_embeddings

        with self._cache_lock:
            if self._taxonomy_embeddings is None:
                embeddings: Dict[str, List[float]] = {}
                for attribute in ATTRIBUTE_ORDER:
                    group = ATTRIBUTE_TO_GROUP[attribute]
                    embeddings[attribute] = _as_finite_vector(
                        self.text_embedder.embed(f"{group}: {attribute}")
                    )
                self._taxonomy_embeddings = embeddings
        return self._taxonomy_embeddings

    def embedding_fallback(self, memory_text: str) -> List[str]:
        if not isinstance(memory_text, str) or not memory_text.strip():
            raise ValueError("Cannot label an empty memory")

        query_vector = _as_finite_vector(self.text_embedder.embed(memory_text))
        taxonomy_embeddings = self._get_taxonomy_embeddings()
        ranked = sorted(
            ATTRIBUTE_ORDER,
            key=lambda attribute: (
                -_cosine_similarity(query_vector, taxonomy_embeddings[attribute]),
                ATTRIBUTE_ORDER.index(attribute),
            ),
        )
        first = ranked[0]
        second = next(
            attribute
            for attribute in ranked[1:]
            if ATTRIBUTE_TO_GROUP[attribute] != ATTRIBUTE_TO_GROUP[first]
        )
        return [first, second]

    def label_extraction_results(
        self,
        extracted_results: List[Optional[Dict[str, Any]]],
        statistics: Dict[str, int],
        logger: Any = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Validate labels in-place and remove facts that cannot be labeled."""
        for result in extracted_results:
            if not result:
                continue
            cleaned_result = result.get("cleaned_result")
            if not isinstance(cleaned_result, list):
                continue

            accepted_entries = []
            for entry in cleaned_result:
                if not isinstance(entry, dict):
                    continue
                statistics["candidate_facts"] += 1
                attributes = validate_attribute_pair(entry.get("attributes"))
                if attributes is not None:
                    entry["attributes"] = attributes
                    statistics["llm_valid_pairs"] += 1
                    accepted_entries.append(entry)
                    continue

                statistics["embedding_fallback_attempts"] += 1
                memory_text = entry.get("fact") or entry.get("relation") or ""
                try:
                    entry["attributes"] = self.embedding_fallback(memory_text)
                except Exception as exc:
                    statistics["embedding_fallback_failures"] += 1
                    statistics["attribute_rejections"] += 1
                    if logger:
                        logger.warning(
                            "Rejecting extracted memory because attribute fallback failed: %s",
                            exc,
                        )
                    continue

                statistics["embedding_fallback_successes"] += 1
                accepted_entries.append(entry)

            result["cleaned_result"] = accepted_entries
        return extracted_results
