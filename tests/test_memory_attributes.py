import math

from lightmem.memory.attributes import (
    ATTRIBUTE_ORDER,
    ATTRIBUTE_TO_GROUP,
    AttributeLabeler,
    add_attribute_instructions,
    empty_attribute_statistics,
    validate_attribute_pair,
)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        if text == "memory about moving for a new job":
            return [1.0, 0.0]
        if text.endswith("Residence & housing"):
            return [1.0, 0.0]
        if text.endswith("Travel & trips"):
            return [0.9, 0.1]
        if text.endswith("Employment & career"):
            return [0.8, 0.2]
        return [0.0, 1.0]


def test_validate_attribute_pair_requires_two_different_groups():
    assert len(ATTRIBUTE_ORDER) == 20
    assert len(set(ATTRIBUTE_ORDER)) == 20
    assert len(set(ATTRIBUTE_TO_GROUP.values())) == 4
    assert validate_attribute_pair([
        "Residence & housing",
        "Employment & career",
    ]) == ["Residence & housing", "Employment & career"]
    assert validate_attribute_pair([
        "Residence & housing",
        "Travel & trips",
    ]) is None
    assert validate_attribute_pair([
        "Residence & housing",
        "Residence & housing",
    ]) is None
    assert validate_attribute_pair(["Residence & housing", "not in taxonomy"]) is None
    assert validate_attribute_pair(["Residence & housing"]) is None


def test_embedding_fallback_skips_second_label_from_same_group_and_caches_taxonomy():
    embedder = FakeEmbedder()
    labeler = AttributeLabeler(embedder)

    first = labeler.embedding_fallback("memory about moving for a new job")
    second = labeler.embedding_fallback("memory about moving for a new job")

    assert first == ["Residence & housing", "Employment & career"]
    assert second == first
    assert ATTRIBUTE_TO_GROUP[first[0]] != ATTRIBUTE_TO_GROUP[first[1]]
    taxonomy_calls = [call for call in embedder.calls if ": " in call]
    assert len(taxonomy_calls) == len(ATTRIBUTE_ORDER)


def test_embedding_fallback_breaks_ties_in_taxonomy_order():
    class TieEmbedder:
        def embed(self, text):
            return [1.0]

    assert AttributeLabeler(TieEmbedder()).embedding_fallback("anything") == [
        "Residence & housing",
        "Employment & career",
    ]


def test_label_extraction_results_uses_llm_pair_then_embedding_fallback():
    labeler = AttributeLabeler(FakeEmbedder())
    statistics = empty_attribute_statistics()
    results = [{
        "cleaned_result": [
            {
                "source_id": 0,
                "fact": "valid",
                "attributes": ["Sleep & energy", "Routines & time"],
            },
            {
                "source_id": 1,
                "fact": "memory about moving for a new job",
                "attributes": ["Residence & housing", "Travel & trips"],
            },
        ]
    }]

    labeled = labeler.label_extraction_results(results, statistics)

    assert labeled[0]["cleaned_result"][0]["attributes"] == [
        "Sleep & energy",
        "Routines & time",
    ]
    assert labeled[0]["cleaned_result"][1]["attributes"] == [
        "Residence & housing",
        "Employment & career",
    ]
    assert statistics == {
        "candidate_facts": 2,
        "llm_valid_pairs": 1,
        "embedding_fallback_attempts": 1,
        "embedding_fallback_successes": 1,
        "embedding_fallback_failures": 0,
        "attribute_rejections": 0,
    }


def test_failed_fallback_rejects_only_the_invalid_fact():
    class FailingEmbedder:
        def embed(self, text):
            return [math.nan]

    labeler = AttributeLabeler(FailingEmbedder())
    statistics = empty_attribute_statistics()
    results = [{
        "cleaned_result": [
            {"source_id": 0, "fact": "bad", "attributes": []},
            {
                "source_id": 1,
                "fact": "valid",
                "attributes": ["Learning & skills", "Interests, tastes & values"],
            },
        ]
    }]

    labeled = labeler.label_extraction_results(results, statistics)

    assert [entry["source_id"] for entry in labeled[0]["cleaned_result"]] == [1]
    assert statistics["embedding_fallback_failures"] == 1
    assert statistics["attribute_rejections"] == 1


def test_prompt_instructions_include_schema_and_taxonomy():
    prompt = add_attribute_instructions("Extract memories.")

    assert "ATTRIBUTE LABELING REQUIREMENT (MANDATORY)" in prompt
    assert '"attributes": ["<primary>", "<secondary>"]' in prompt
    for attribute in ATTRIBUTE_ORDER:
        assert attribute in prompt
