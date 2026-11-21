from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import QwenConfig

THINK_RE = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)
CONTEXT_ERROR_RE = re.compile(
    r"maximum context length is (?P<limit>\d+) tokens.*request has "
    r"(?P<input>\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)


class ThinkingOutputError(RuntimeError):
    pass


def adjusted_max_tokens_for_context_error(
    error: Exception, requested_max_tokens: int, safety_margin: int = 64
) -> int | None:
    match = CONTEXT_ERROR_RE.search(str(error))
    if not match:
        return None
    remaining = int(match.group("limit")) - int(match.group("input"))
    adjusted = remaining - safety_margin
    if adjusted < 64 or adjusted >= requested_max_tokens:
        return None
    return adjusted


def assert_no_thinking(content: str, message: Any = None) -> None:
    if THINK_RE.search(content or ""):
        raise ThinkingOutputError("Qwen response contained a <think> marker")
    if message is not None:
        for field in ("reasoning", "reasoning_content"):
            value = getattr(message, field, None)
            if value and str(value).strip():
                raise ThinkingOutputError(f"Qwen response contained nonempty {field}")


def normalize_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


@dataclass
class GenerationResult:
    content: str
    elapsed_seconds: float
    usage: dict[str, int]
    attempts: int
    no_thinking_verified: bool = True


class NoThinkingQwenClient:
    """OpenAI-compatible Qwen client that hard-disables and verifies thinking."""

    def __init__(self, config: QwenConfig, max_retries: int = 3):
        if config.thinking:
            raise ValueError(
                "NoThinkingQwenClient refuses thinking-enabled configuration"
            )
        self.config = config
        self.max_retries = max_retries
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> GenerationResult:
        last_error: Exception | None = None
        active_max_tokens = max_tokens or self.config.answer_max_tokens
        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "seed": self.config.seed,
                    "max_tokens": active_max_tokens,
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = message.content or ""
                assert_no_thinking(content, message)
                if not content.strip():
                    raise ValueError("Qwen returned empty content")
                return GenerationResult(
                    content=content,
                    elapsed_seconds=time.perf_counter() - started,
                    usage=normalize_usage(response),
                    attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
                adjusted = adjusted_max_tokens_for_context_error(
                    exc, active_max_tokens
                )
                if adjusted is not None:
                    active_max_tokens = adjusted
                    continue
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(
            f"Qwen generation failed after {self.max_retries} attempts"
        ) from last_error

    def preflight(self) -> None:
        plain = self.generate(
            [{"role": "user", "content": "Reply with exactly: ready"}],
            max_tokens=16,
        )
        if "ready" not in plain.content.lower():
            raise RuntimeError(f"Unexpected Qwen preflight response: {plain.content!r}")
        structured = self.generate(
            [{"role": "user", "content": 'Return JSON: {"ready": true}'}],
            max_tokens=32,
            response_format={"type": "json_object"},
        )
        import json

        if json.loads(structured.content).get("ready") is not True:
            raise RuntimeError("Qwen structured-output preflight failed")


class _NoThinkingCompletionsProxy:
    def __init__(self, inner: Any):
        self._inner = inner

    def create(self, *args: Any, **kwargs: Any) -> Any:
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
        template_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = template_kwargs
        response = self._inner.create(*args, extra_body=extra_body, **kwargs)
        message = response.choices[0].message
        assert_no_thinking(message.content or "", message)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _NoThinkingChatProxy:
    def __init__(self, inner: Any):
        self._inner = inner
        self.completions = _NoThinkingCompletionsProxy(inner.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _NoThinkingOpenAIProxy:
    def __init__(self, inner: Any):
        self._inner = inner
        self.chat = _NoThinkingChatProxy(inner.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_openai_client_no_thinking(client: Any) -> Any:
    """Force and validate no-thinking on every Chat Completions request."""
    return _NoThinkingOpenAIProxy(client)


def answer_messages(
    query: str, memories: list[str], dimension: str
) -> list[dict[str, str]]:
    if dimension == "IPA":
        system = (
            "You are a helpful assistant. Review the retrieved memories about the user, "
            "then respond to the user's latest query directly."
        )
        query_label = "Latest Query"
    else:
        system = (
            "You are a helpful assistant. Review the retrieved memories about the user, "
            "then accurately answer the question."
        )
        query_label = "Question"
    memory_text = "\n".join(memories) if memories else "(No memories retrieved.)"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"[Retrieved Memories]\n{memory_text}\n\n[{query_label}]\n{query}",
        },
    ]
