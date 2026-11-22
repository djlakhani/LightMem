import json
import logging
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from lightmem.memory.taxonomy import (
    TaxonomyValidationError,
    repair_extracted_result_labels,
    taxonomy_digest,
    validate_category_labels,
)
from lightmem.memory.lightmem import LightMemory
from lightmem.factory.memory_manager.openai import OpenaiManager
from stale_lightmem.dependency_resolution import (
    anchor_selection_response_format,
    build_anchor_selection_messages,
    build_final_answer_messages,
    build_resolution_messages,
    generate_dependency_resolution,
    generate_validated_json,
    parse_anchor_selection,
    parse_resolution_response,
    resolution_response_format,
)
from stale_lightmem.backend import LightMemBackend
from stale_lightmem.qwen import GenerationResult


LABELS = ["residence_housing", "goals_commitments"]


def _memory(memory_id: str, timestamp: float, text: str):
    return {
        "id": memory_id,
        "score": 1.0,
        "payload": {
            "memory": text,
            "time_stamp": f"2025-01-{int(timestamp):02d}T10:00:00.000",
            "float_time_stamp": timestamp,
            "category_labels": LABELS,
        },
    }


def _candidate(memory_id: str, timestamp: float, text: str):
    item = _memory(memory_id, timestamp, text)
    item.update(
        {
            "category_labels": [
                {
                    "id": "residence_housing",
                    "name": "Residence & housing",
                    "group": "living_logistics",
                },
                {
                    "id": "goals_commitments",
                    "name": "Goals & commitments",
                    "group": "work_finance",
                },
            ],
            "matching_labels": [
                {
                    "id": "residence_housing",
                    "name": "Residence & housing",
                    "group": "living_logistics",
                }
            ],
            "selection_source": "semantic",
        }
    )
    return item


def test_taxonomy_requires_two_labels_from_different_groups():
    assert validate_category_labels(LABELS) == LABELS
    assert validate_category_labels(
        ["Residence Housing", "goalscommitments"]
    ) == LABELS
    with pytest.raises(TaxonomyValidationError, match="different parent"):
        validate_category_labels(["residence_housing", "travel_trips"])
    with pytest.raises(TaxonomyValidationError, match="exactly two"):
        validate_category_labels(["residence_housing"])
    assert len(taxonomy_digest()) == 64


def test_extraction_repairs_invalid_same_group_labels_once():
    manager = object.__new__(OpenaiManager)
    manager.taxonomy_labeling = True
    responses = iter(
        [
            json.dumps(
                {
                    "data": [
                        {
                            "source_id": 0,
                            "fact": "The user moved homes.",
                            "labels": ["residence_housing", "social_other"],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "data": [
                        {
                            "source_id": 0,
                            "fact": "The user moved homes.",
                            "labels": LABELS,
                        }
                    ]
                }
            ),
        ]
    )
    calls = []

    def generate_response(messages, response_format):
        calls.append(messages)
        return next(responses), {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }

    manager.generate_response = generate_response
    results = manager.meta_text_extract(
        extract_list=[
            [
                [
                    {
                        "role": "user",
                        "content": "I moved homes.",
                        "sequence_number": 0,
                        "time_stamp": "2025-01-01T10:00:00.000",
                        "weekday": "Wed",
                    }
                ]
            ]
        ],
        messages_use="user_only",
        topic_id_mapping=[[0]],
    )
    assert len(calls) == 2
    assert "Parent buckets are not selectable" in calls[1][-1]["content"]
    assert results[0]["cleaned_result"][0]["labels"] == LABELS
    assert results[0]["usage"]["total_tokens"] == 4


def test_repeated_bad_model_labels_use_local_semantic_fallback():
    manager = object.__new__(OpenaiManager)
    manager.taxonomy_labeling = True
    invalid = json.dumps(
        {
            "data": [
                {
                    "source_id": 0,
                    "fact": "The user moved to a new apartment for work.",
                    "labels": ["living_logistics", "work_finance"],
                }
            ]
        }
    )
    manager.generate_response = lambda messages, response_format: (
        invalid,
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    results = manager.meta_text_extract(
        extract_list=[
            [
                [
                    {
                        "role": "user",
                        "content": "I moved to a new apartment for work.",
                        "sequence_number": 0,
                        "time_stamp": "2025-01-01T10:00:00.000",
                        "weekday": "Wed",
                    }
                ]
            ]
        ],
        messages_use="user_only",
        topic_id_mapping=[[0]],
    )
    assert results[0]["taxonomy_validation_error"]

    class FakeEmbedder:
        def embed(self, text):
            return [float("housing" in text.lower()), float("work" in text.lower()), 1.0]

    repair_extracted_result_labels(results, FakeEmbedder())
    repaired = results[0]["cleaned_result"][0]
    assert validate_category_labels(repaired["labels"]) == repaired["labels"]
    assert repaired["taxonomy_label_status"] == "semantic_fallback"
    assert repaired["taxonomy_original_labels"] == [
        "living_logistics",
        "work_finance",
    ]


def test_anchor_prompt_hides_taxonomy_and_parser_requires_exact_ids():
    retrieval = [_memory(f"id-{index}", index, f"memory {index}") for index in range(1, 6)]
    messages = build_anchor_selection_messages("What changed?", retrieval, "SR", 4)
    rendered = json.dumps(messages)
    assert "category_labels" not in rendered
    assert "residence_housing" not in rendered
    parsed = parse_anchor_selection(
        '{"anchor_ids":["id-1","id-2","id-3","id-4"]}',
        {f"id-{index}" for index in range(1, 6)},
        4,
    )
    assert parsed["anchor_ids"] == ["id-1", "id-2", "id-3", "id-4"]
    with pytest.raises(ValueError, match="not in retrieval"):
        parse_anchor_selection(
            '{"anchor_ids":["id-1","id-2","id-3","invented"]}',
            {f"id-{index}" for index in range(1, 6)},
            4,
        )
    schema = anchor_selection_response_format(
        4, {f"id-{index}" for index in range(1, 6)}
    )
    anchor_items = schema["json_schema"]["schema"]["properties"]["anchor_ids"]
    assert anchor_items["uniqueItems"] is True
    assert set(anchor_items["items"]["enum"]) == {
        f"id-{index}" for index in range(1, 6)
    }


def test_resolution_requires_full_candidate_audit_and_distills_summary():
    anchors = [_memory("anchor", 1, "The user plans to stay five months.")]
    candidates = {
        "anchor": [
            _candidate("new", 2, "The user accepted an eighteen-month assignment."),
            _candidate("other", 3, "The user bought a lamp."),
        ]
    }
    messages = build_resolution_messages("How long?", anchors, candidates, 3)
    assert "matching_labels" in messages[1]["content"]
    response = json.dumps(
        {
            "anchor_resolutions": [
                {
                    "anchor_id": "anchor",
                    "candidate_verdicts": [
                        {"memory_id": "new", "relation": "invalidates"},
                        {"memory_id": "other", "relation": "none"},
                    ],
                    "selected_changes": [
                        {
                            "memory_id": "new",
                            "relation": "invalidates",
                            "why": "The later assignment replaces the five-month plan.",
                        }
                    ],
                    "resolution_summary": (
                        "The current stay is eighteen months, replacing the old plan."
                    ),
                }
            ]
        }
    )
    parsed = parse_resolution_response(response, anchors, candidates, 3)
    final_messages = build_final_answer_messages(
        "How long?", "SR", anchors, candidates, parsed
    )
    final_payload = json.loads(final_messages[1]["content"])
    package = final_payload["resolved_evidence"][0]
    assert package["category_resolution"].startswith("The current stay")
    assert package["selected_later_states"][0]["relation"] == "invalidates"

    schema = resolution_response_format(anchors, candidates, 3)
    resolution_schema = schema["json_schema"]["schema"]["properties"][
        "anchor_resolutions"
    ]
    assert resolution_schema["required"] == ["anchor"]
    verdict_schema = resolution_schema["properties"]["anchor"]["properties"][
        "candidate_verdicts"
    ]
    assert verdict_schema["required"] == ["new", "other"]
    assert verdict_schema["additionalProperties"] is False

    incomplete = response.replace(
        ', {"memory_id": "other", "relation": "none"}', ""
    )
    with pytest.raises(ValueError, match="skipped candidates"):
        parse_resolution_response(incomplete, anchors, candidates, 3)


def test_truncated_structured_output_is_retried_even_if_json_is_parseable():
    class TruncatingQwen:
        def __init__(self):
            self.calls = 0

        def generate(self, messages, max_tokens, response_format=None):
            self.calls += 1
            return GenerationResult(
                content='{"ok": true}',
                elapsed_seconds=0.1,
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                attempts=1,
                finish_reason="length" if self.calls == 1 else "stop",
            )

    qwen = TruncatingQwen()
    generated = generate_validated_json(
        qwen,
        [{"role": "user", "content": "return JSON"}],
        lambda content: json.loads(content),
        max_tokens=32,
    )
    assert qwen.calls == 2
    assert [attempt["status"] for attempt in generated["attempts"]] == [
        "invalid",
        "valid",
    ]
    assert "truncated" in generated["attempts"][0]["error"]


def test_malformed_batched_resolution_falls_back_to_each_anchor():
    anchors = [
        _memory("anchor-1", 1, "The user plans to stay five months."),
        _memory("anchor-2", 1, "The user works remotely."),
    ]
    candidates = {
        "anchor-1": [_candidate("future-1", 2, "The stay is now eighteen months.")],
        "anchor-2": [_candidate("future-2", 2, "The user now works on site.")],
    }

    class MalformedBatchQwen:
        def __init__(self):
            self.calls = 0
            self.response_formats = []

        def generate(self, messages, max_tokens, response_format=None):
            self.calls += 1
            self.response_formats.append(response_format)
            if self.calls <= 2:
                content = '{"anchor_resolutions": {'
            else:
                group = json.loads(messages[1]["content"])["anchor_groups"][0]
                anchor_id = group["anchor"]["id"]
                candidate_id = group[
                    "future_states_within_matching_categories"
                ][0]["id"]
                content = json.dumps(
                    {
                        "anchor_resolutions": {
                            anchor_id: {
                                "candidate_verdicts": {
                                    candidate_id: "updates"
                                },
                                "selected_changes": [
                                    {
                                        "memory_id": candidate_id,
                                        "relation": "updates",
                                        "why": (
                                            "The later state directly changes the "
                                            "corresponding earlier state."
                                        ),
                                    }
                                ],
                                "resolution_summary": (
                                    "The later state is now current and replaces the "
                                    "corresponding anchor detail."
                                ),
                            }
                        }
                    }
                )
            return GenerationResult(
                content=content,
                elapsed_seconds=0.1,
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                attempts=1,
                finish_reason="stop",
            )

    qwen = MalformedBatchQwen()
    generated = generate_dependency_resolution(
        qwen,
        "What is current?",
        anchors,
        candidates,
        3,
        max_tokens=2048,
    )
    assert generated["mode"] == "per_anchor_fallback"
    assert qwen.calls == 4
    assert [
        resolution["anchor_id"]
        for resolution in generated["parsed"]["anchor_resolutions"]
    ] == ["anchor-1", "anchor-2"]
    assert all(
        response_format["type"] == "json_schema"
        for response_format in qwen.response_formats
    )


def test_all_none_summary_reaches_final_answerer():
    anchors = [_memory("anchor", 1, "The user plans to stay five months.")]
    candidates = {"anchor": [_candidate("later", 2, "The user bought a lamp.")]}
    response = json.dumps(
        {
            "anchor_resolutions": [
                {
                    "anchor_id": "anchor",
                    "candidate_verdicts": [
                        {"memory_id": "later", "relation": "none"}
                    ],
                    "selected_changes": [],
                    "resolution_summary": (
                        "The later purchase does not change the stated duration of the stay."
                    ),
                }
            ]
        }
    )
    parsed = parse_resolution_response(response, anchors, candidates, 3)
    final = build_final_answer_messages("How long?", "SR", anchors, candidates, parsed)
    assert "does not change" in final[1]["content"]
    assert "The user bought a lamp" not in final[1]["content"]


def test_future_candidates_are_eight_semantic_plus_four_newest_unique():
    semantic = [
        _memory(f"semantic-{index}", index + 2, f"semantic {index}")
        for index in range(8)
    ]
    recent = [
        _memory(f"recent-{index}", index + 20, f"recent {index}")
        for index in range(6)
    ]
    observed = {}

    class FakeRetriever:
        def search(self, **kwargs):
            observed.update(kwargs)
            return semantic

        def get_all(self, **kwargs):
            return semantic + recent

    backend = object.__new__(LightMemBackend)
    backend.memory = SimpleNamespace(
        text_embedder=SimpleNamespace(embed=lambda text: [1.0, 0.0]),
        embedding_retriever=FakeRetriever(),
    )
    anchor = _memory("anchor", 1, "anchor state")
    selected = backend.future_candidates(anchor, semantic_limit=8, recent_limit=4)
    assert [item["id"] for item in selected[:8]] == [
        f"semantic-{index}" for index in range(8)
    ]
    assert [item["id"] for item in selected[8:]] == [
        "recent-5",
        "recent-4",
        "recent-3",
        "recent-2",
    ]
    assert len({item["id"] for item in selected}) == 12
    assert observed["filters"] == {
        "float_time_stamp": {"gt": 1.0},
        "category_labels": LABELS,
    }


def test_qdrant_filters_strictly_later_and_any_matching_label():
    client = QdrantClient(":memory:")
    store = Qdrant(
        QdrantConfig(
            collection_name="dependency-filter-test",
            embedding_model_dims=2,
            client=client,
            path="/tmp/unused",
        )
    )
    store.insert(
        vectors=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
        ids=[1, 2, 3],
        payloads=[
            {"float_time_stamp": 1.0, "category_labels": LABELS},
            {
                "float_time_stamp": 2.0,
                "category_labels": [
                    "residence_housing",
                    "emotional_wellbeing",
                ],
            },
            {
                "float_time_stamp": 3.0,
                "category_labels": ["sleep_energy", "social_participation"],
            },
        ],
    )
    hits = store.search(
        [1.0, 0.0],
        limit=8,
        filters={
            "float_time_stamp": {"gt": 1.0},
            "category_labels": LABELS,
        },
        return_full=True,
    )
    assert [hit["id"] for hit in hits] == [2]
    client.close()


def test_consolidation_relabels_and_reembeds_updated_memory():
    entries = [
        {
            "id": "old",
            "vector": [1.0, 0.0],
            "payload": {
                "memory": "The user plans to stay five months.",
                "category_labels": LABELS,
                "update_queue": [],
            },
        },
        {
            "id": "new",
            "vector": [0.9, 0.1],
            "payload": {
                "memory": "The user accepted an eighteen-month assignment.",
                "category_labels": LABELS,
                "update_queue": [{"id": "old", "score": 0.95}],
            },
        },
    ]
    updates = []
    embedded = []

    class FakeRetriever:
        def get_all(self):
            return entries

        def update(self, vector_id, vector, payload):
            updates.append((vector_id, vector, payload))

        def delete(self, vector_id):
            raise AssertionError(f"unexpected delete: {vector_id}")

    memory = object.__new__(LightMemory)
    memory.config = SimpleNamespace(taxonomy_labeling=True)
    memory.embedding_retriever = FakeRetriever()
    memory.text_embedder = SimpleNamespace(
        embed=lambda text: embedded.append(text) or [0.2, 0.8]
    )
    memory.manager = SimpleNamespace(
        _call_update_llm=lambda prompt, target, candidates: {
            "action": "update",
            "new_memory": "The user now plans to stay eighteen months.",
            "labels": ["residence_housing", "employment_career"],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
    )
    memory.logger = logging.getLogger("test-consolidation")
    memory.token_stats = {
        "update_calls": 0,
        "update_prompt_tokens": 0,
        "update_completion_tokens": 0,
        "update_total_tokens": 0,
    }
    memory.offline_update_all_entries(score_threshold=0.8, max_workers=1)
    assert embedded == ["The user now plans to stay eighteen months."]
    assert updates[0][0] == "old"
    assert updates[0][1] == [0.2, 0.8]
    assert updates[0][2]["category_labels"] == [
        "residence_housing",
        "employment_career",
    ]
