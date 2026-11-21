from types import SimpleNamespace

import pytest
from stale_lightmem.qwen import (
    NoThinkingQwenClient,
    ThinkingOutputError,
    adjusted_max_tokens_for_context_error,
    assert_no_thinking,
    wrap_openai_client_no_thinking,
)


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


def test_context_error_reduces_reserved_completion_tokens():
    error = RuntimeError(
        "maximum context length is 8192 tokens and your request has 6561 input tokens"
    )
    assert adjusted_max_tokens_for_context_error(error, 2048) == 1567
    assert adjusted_max_tokens_for_context_error(RuntimeError("other"), 2048) is None


def test_qwen_retries_context_error_with_smaller_completion_budget():
    observed = []

    class Completions:
        def create(self, **kwargs):
            observed.append(kwargs["max_tokens"])
            if len(observed) == 1:
                raise RuntimeError(
                    "maximum context length is 8192 tokens and your request has "
                    "6561 input tokens"
                )
            message = SimpleNamespace(
                content='{"ok": true}', reasoning=None, reasoning_content=None
            )
            usage = SimpleNamespace(
                prompt_tokens=6561, completion_tokens=4, total_tokens=6565
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], usage=usage
            )

    qwen = object.__new__(NoThinkingQwenClient)
    qwen.config = SimpleNamespace(
        model="qwen",
        temperature=0.0,
        top_p=1.0,
        seed=42,
        answer_max_tokens=512,
    )
    qwen.max_retries = 3
    qwen.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = qwen.generate(
        [{"role": "user", "content": "test"}],
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    assert result.attempts == 2
    assert observed == [2048, 1567]
