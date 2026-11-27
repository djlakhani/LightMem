import time

import pytest

from stale_lightmem.backend import build_future_candidates
from stale_lightmem.qwen import (
    GenerationResult,
    answer_messages,
    build_anchor_contexts,
    filter_future_updates,
    parse_future_updates,
)


def _memory(memory_id, timestamp, attributes, vector, text=None):
    return {
        "id": memory_id,
        "vector": vector,
        "payload": {
            "time_stamp": f"2026-01-{int(timestamp):02d}",
            "float_time_stamp": float(timestamp),
            "attributes": attributes,
            "memory": text or memory_id,
        },
    }


def test_future_candidates_use_strict_time_label_similarity_and_recency():
    anchor_point = _memory("anchor", 10, ["label-a", "label-b"], [1.0, 0.0])
    anchor = {"id": "anchor", "payload": anchor_point["payload"]}
    points = [
        anchor_point,
        _memory("older", 9, ["label-a", "other"], [1.0, 0.0]),
        _memory("same-time", 10, ["label-a", "other"], [1.0, 0.0]),
        _memory("wrong-label", 30, ["other", "different"], [1.0, 0.0]),
        _memory("similar-1", 11, ["label-a", "other"], [1.0, 0.0]),
        _memory("similar-2", 12, ["label-b", "other"], [0.9, 0.1]),
        _memory("recent-1", 40, ["label-a", "other"], [0.0, 1.0]),
        _memory("later-anchor", 35, ["label-b", "other"], [-1.0, 0.0]),
        _memory("remaining", 20, ["label-a", "other"], [0.5, 0.5]),
    ]
    result = build_future_candidates(
        anchor, points, similarity_limit=2, recency_limit=2
    )
    assert result["eligible_count"] == 5
    assert [item["id"] for item in result["candidates"]] == [
        "similar-1",
        "similar-2",
        "recent-1",
        "later-anchor",
    ]
    assert [item["candidate_source"] for item in result["candidates"]] == [
        "similarity",
        "similarity",
        "recency",
        "recency",
    ]
    assert result["candidates"][1]["matched_attributes"] == ["label-b"]


def test_future_candidates_are_capped_at_eight_similarity_plus_four_recency():
    anchor_point = _memory("anchor", 1, ["label-a", "label-b"], [1.0, 0.0])
    anchor = {"id": "anchor", "payload": anchor_point["payload"]}
    points = [anchor_point] + [
        _memory(
            f"future-{index:02d}",
            index + 2,
            ["label-a", "other"],
            [1.0, float(index + 1)],
        )
        for index in range(20)
    ]
    result = build_future_candidates(anchor, points)
    assert result["eligible_count"] == 20
    assert len(result["candidates"]) == 12
    assert (
        sum(item["candidate_source"] == "similarity" for item in result["candidates"])
        == 8
    )
    assert (
        sum(item["candidate_source"] == "recency" for item in result["candidates"]) == 4
    )
    assert len({item["id"] for item in result["candidates"]}) == 12


def _candidate(memory_id):
    return {
        "id": memory_id,
        "score": 0.8,
        "matched_attributes": ["label-a"],
        "candidate_source": "similarity",
        "source_rank": 1,
        "payload": {
            "time_stamp": "2026-01-20",
            "float_time_stamp": 20.0,
            "attributes": ["label-a", "other"],
            "memory": f"future {memory_id}",
        },
    }


def _anchor(memory_id="anchor"):
    return {
        "id": memory_id,
        "payload": {
            "time_stamp": "2026-01-10",
            "float_time_stamp": 10.0,
            "attributes": ["label-a", "label-b"],
            "memory": f"anchor {memory_id}",
        },
    }


class FakeQwen:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, messages, max_tokens, response_format):
        return GenerationResult(
            content=next(self.responses),
            elapsed_seconds=0.1,
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            attempts=1,
        )


def test_future_update_parser_accepts_empty_and_normalizes_explanations():
    candidates = [_candidate("future-1"), _candidate("future-2")]
    assert parse_future_updates('{"updates": []}', candidates) == []
    parsed = parse_future_updates(
        '{"updates":[{"memory_id":"future-2","relationship":"update",'
        '"explanation":"Later fact\\n  changes it."}]}',
        candidates,
    )
    assert parsed == [
        {
            "memory_id": "future-2",
            "relationship": "update",
            "explanation": "Later fact changes it.",
        }
    ]


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (
            (
                '{"updates":[{"memory_id":"invented","relationship":"update",'
                '"explanation":"x"}]}'
            ),
            "invented",
        ),
        (
            (
                '{"updates":[{"memory_id":"future-1","relationship":"update",'
                '"explanation":"x"},{"memory_id":"future-1",'
                '"relationship":"append","explanation":"y"}]}'
            ),
            "duplicate",
        ),
        (
            (
                '{"updates":[{"memory_id":"future-1","relationship":"update",'
                '"explanation":""}]}'
            ),
            "nonempty",
        ),
        (
            (
                '{"updates":[{"memory_id":"future-1","relationship":"bad",'
                '"explanation":"x"}]}'
            ),
            "relationship",
        ),
        ('{"updates":[{"memory_id":"future-1"}]}', "only memory_id"),
    ],
)
def test_future_update_parser_rejects_malformed_selections(content, error):
    with pytest.raises(ValueError, match=error):
        parse_future_updates(content, [_candidate("future-1")])


def test_future_filter_repairs_once_and_accepts_the_repair():
    search = {
        "anchor_id": "anchor",
        "eligible_count": 1,
        "candidates": [_candidate("future-1")],
        "skip_reason": None,
    }
    qwen = FakeQwen(
        [
            '{"updates":"bad"}',
            (
                '{"updates":[{"memory_id":"future-1","relationship":"update",'
                '"explanation":"It changes."}]}'
            ),
        ]
    )
    selected, _, result = filter_future_updates(qwen, _anchor(), search)
    assert [item["id"] for item in selected] == ["future-1"]
    assert result.method == "llm"
    assert result.fallback_used is False
    assert result.llm_attempts == 2
    assert result.attempts[0].validation_error
    assert result.attempts[1].validation_error is None


def test_future_filter_repairs_once_then_uses_no_update_fallback():
    search = {
        "anchor_id": "anchor",
        "eligible_count": 1,
        "candidates": [_candidate("future-1")],
        "skip_reason": None,
    }
    qwen = FakeQwen(['{"updates":[{"bad":true}]}', '{"updates":"bad"}'])
    selected, _, result = filter_future_updates(qwen, _anchor(), search)
    assert selected == []
    assert result.method == "no_updates_fallback"
    assert result.fallback_used is True
    assert result.llm_attempts == 2
    assert len(result.attempts) == 2
    assert all(attempt.validation_error for attempt in result.attempts)


def test_future_filter_skips_qwen_when_no_candidates_exist():
    selected, prompt, result = filter_future_updates(
        FakeQwen([]),
        _anchor(),
        {
            "anchor_id": "anchor",
            "eligible_count": 0,
            "candidates": [],
            "skip_reason": None,
        },
    )
    assert selected == []
    assert prompt == []
    assert result.method == "no_candidates"
    assert result.llm_attempts == 0


class ConcurrentFakeQwen:
    def generate(self, messages, max_tokens, response_format):
        prompt = messages[1]["content"]
        anchor_id = prompt.split("Memory ID: ", 1)[1].splitlines()[0]
        if anchor_id == "anchor-1":
            time.sleep(0.02)
        return GenerationResult(
            content='{"updates":[]}',
            elapsed_seconds=0.01,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            attempts=1,
        )


def test_parallel_anchor_filtering_preserves_anchor_order_and_final_context():
    anchors = [_anchor("anchor-1"), _anchor("anchor-2")]
    searches = [
        {
            "anchor_id": anchor["id"],
            "eligible_count": 1,
            "candidates": [_candidate(f"future-{anchor['id']}")],
            "skip_reason": None,
        }
        for anchor in anchors
    ]
    contexts = build_anchor_contexts(
        ConcurrentFakeQwen(), anchors, searches, max_workers=2
    )
    assert [context["anchor"]["id"] for context in contexts] == [
        "anchor-1",
        "anchor-2",
    ]
    contexts[0]["selected_updates"] = [
        {
            **_candidate("future-anchor-1"),
            "update_relationship": "update",
            "update_explanation": "This changes the anchor.",
        }
    ]
    messages = answer_messages("What is current?", contexts, "SR")
    prompt = messages[1]["content"]
    assert "Memory ID: anchor-1" in prompt
    assert "Memory ID: future-anchor-1" in prompt
    assert "This changes the anchor." in prompt
    assert "Relationship: update" in prompt
    assert "future-anchor-2" not in prompt
