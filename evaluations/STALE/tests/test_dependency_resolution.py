import json
import logging
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from lightmem.memory.taxonomy import (
    TaxonomyValidationError,
    taxonomy_digest,
    validate_category_labels,
)
from lightmem.memory.lightmem import LightMemory
from lightmem.factory.memory_manager.openai import OpenaiManager
from stale_lightmem.dependency_resolution import (
    build_anchor_selection_messages,
    build_final_answer_messages,
    build_resolution_messages,
    parse_anchor_selection,
    parse_resolution_response,
)
from stale_lightmem.backend import LightMemBackend


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
                            "labels": ["residence_housing", "travel_trips"],
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
    assert results[0]["cleaned_result"][0]["labels"] == LABELS
    assert results[0]["usage"]["total_tokens"] == 4


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

    incomplete = response.replace(
        ', {"memory_id": "other", "relation": "none"}', ""
    )
    with pytest.raises(ValueError, match="skipped candidates"):
        parse_resolution_response(incomplete, anchors, candidates, 3)


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
