from types import SimpleNamespace

import pytest
from stale_lightmem.qwen import (
    ThinkingOutputError,
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
