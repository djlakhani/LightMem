from types import SimpleNamespace

import pytest

from stale_lightmem.qwen import (
    GenerationResult,
    ThinkingOutputError,
    answer_messages,
    assert_no_thinking,
    evidence_selection_messages,
    parse_evidence_selection,
    select_relevant_memories,
    wrap_openai_client_no_thinking,
)


def _retrieval():
    return [
        {
            "id": f"memory-{index}",
            "score": 1.0 - index / 10,
            "payload": {
                "time_stamp": f"2026-01-0{index + 1}",
                "attributes": ["Residence & housing", "Goals & commitments"],
                "memory": f"memory text {index}",
            },
        }
        for index in range(5)
    ]


def test_no_thinking_accepts_plain_content():
    assert_no_thinking("ready", SimpleNamespace(reasoning=None, reasoning_content=None))


@pytest.mark.parametrize("content", ["<think>secret</think>answer", "<THINK>bad"])
def test_no_thinking_rejects_markers(content):
    with pytest.raises(ThinkingOutputError):
        assert_no_thinking(content)


def test_no_thinking_rejects_reasoning_field():
    with pytest.raises(ThinkingOutputError):
        assert_no_thinking("answer", SimpleNamespace(reasoning="hidden"))


def test_proxy_forces_template_flag_on_memory_calls():
    observed = {}

    class Completions:
        def create(self, **kwargs):
            observed.update(kwargs)
            message = SimpleNamespace(
                content="{}", reasoning=None, reasoning_content=None
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    wrapped = wrap_openai_client_no_thinking(client)
    wrapped.chat.completions.create(model="qwen", messages=[])
    assert observed["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_evidence_selection_prompt_includes_ids_labels_and_question():
    messages = evidence_selection_messages("What changed?", _retrieval())
    user_prompt = messages[1]["content"]
    assert "What changed?" in user_prompt
    assert "memory-0" in user_prompt
    assert "Residence & housing" in user_prompt
    assert "Select exactly 4" in user_prompt


def test_evidence_selection_requires_four_valid_unique_ids():
    retrieval = _retrieval()
    selected = parse_evidence_selection(
        '{"memory_ids":["memory-3","memory-1","memory-0","memory-2"]}',
        retrieval,
    )
    assert selected == ["memory-3", "memory-1", "memory-0", "memory-2"]

    with pytest.raises(ValueError, match="exactly 4"):
        parse_evidence_selection('{"memory_ids":["memory-0"]}', retrieval)
    with pytest.raises(ValueError, match="duplicate"):
        parse_evidence_selection(
            '{"memory_ids":["memory-0","memory-0","memory-1","memory-2"]}',
            retrieval,
        )
    with pytest.raises(ValueError, match="invented"):
        parse_evidence_selection(
            '{"memory_ids":["memory-0","memory-1","memory-2","invented"]}',
            retrieval,
        )


def test_evidence_selection_uses_all_memories_when_fewer_than_four_exist():
    retrieval = _retrieval()[:2]
    messages = evidence_selection_messages("What changed?", retrieval)
    assert "Select exactly 2" in messages[1]["content"]
    assert parse_evidence_selection(
        '{"memory_ids":["memory-1","memory-0"]}', retrieval
    ) == ["memory-1", "memory-0"]


def test_answer_prompt_contains_only_selected_memory_payloads():
    selected = [_retrieval()[3], _retrieval()[1]]
    messages = answer_messages("What changed?", selected, "SR")
    user_prompt = messages[1]["content"]
    assert "memory-3" in user_prompt
    assert "memory text 3" in user_prompt
    assert "Residence & housing" in user_prompt
    assert "memory-0" not in user_prompt


class FakeSelectionQwen:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, messages, max_tokens, response_format):
        return GenerationResult(
            content=next(self.responses),
            elapsed_seconds=0.1,
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            attempts=1,
        )


def test_evidence_selection_repairs_once_after_invalid_response():
    qwen = FakeSelectionQwen(
        [
            '{"memory_ids":[]}',
            '{"memory_ids":["memory-3","memory-1","memory-0","memory-2"]}',
        ]
    )
    selected, _, result = select_relevant_memories(qwen, "What changed?", _retrieval())
    assert [item["id"] for item in selected] == [
        "memory-3",
        "memory-1",
        "memory-0",
        "memory-2",
    ]
    assert result.method == "llm"
    assert result.fallback_used is False
    assert result.llm_attempts == 2
    assert result.attempts[0].validation_error
    assert result.attempts[1].validation_error is None
    assert result.usage["total_tokens"] == 10


def test_evidence_selection_falls_back_to_embedding_order_after_retry():
    qwen = FakeSelectionQwen(['{"memory_ids":[]}', '{"memory_ids":["invented"]}'])
    selected, _, result = select_relevant_memories(qwen, "What changed?", _retrieval())
    assert [item["id"] for item in selected] == [
        "memory-0",
        "memory-1",
        "memory-2",
        "memory-3",
    ]
    assert result.method == "embedding_rank_fallback"
    assert result.fallback_used is True
    assert (
        result.fallback_reason == "Evidence selection must contain exactly 4 memory IDs"
    )
    assert result.llm_attempts == 2
    assert all(attempt.validation_error for attempt in result.attempts)
