"""OpenAIProvider — LLMProvider implementation for OpenAI-compatible APIs.

Works with OpenAI, OpenRouter, and any API that speaks the OpenAI
chat-completions format.  Requires the ``openai`` package::

    pip install agent-core[openai]

Usage::

    from openai import OpenAI
    from agent_core.providers.openai import OpenAIProvider

    # OpenRouter
    provider = OpenAIProvider(
        client=OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="<OPENROUTER_API_KEY>",
        ),
    )

    # Standard OpenAI
    provider = OpenAIProvider(client=OpenAI())
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Callable

from agent_core.providers.types import ParsedResponse, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Return a JSON-compatible representation while preserving structure."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either SDK objects or plain dicts."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_cache_config(cache_config: Any | None) -> dict[str, Any]:
    """Normalize provider cache hints into a plain dict.

    Gemini's explicit cache manager passes keys like ``cache_name`` and
    ``contents_offset``. Those are intentionally ignored by this provider.
    OpenRouter-specific keys are consumed by ``_apply_request_options``.
    """
    if cache_config is None:
        return {}
    if hasattr(cache_config, "to_openai_cache_config"):
        cache_config = cache_config.to_openai_cache_config()
    if isinstance(cache_config, dict):
        return dict(cache_config)
    return {}


class _StreamedOpenAIResponse:
    """Minimal chat-completion response consumed by ``parse_response``."""

    def __init__(
        self,
        *,
        message: Any,
        usage: Any | None,
        streamed_text: bool,
    ) -> None:
        self.choices = [SimpleNamespace(message=message)]
        self.usage = usage
        self._agent_core_streamed_text = streamed_text


class OpenAIProvider:
    """LLMProvider for OpenAI-compatible chat-completion APIs.

    Args:
        client: An ``openai.OpenAI`` (or compatible) client instance.
        preserve_reasoning: If ``True``, capture ``reasoning_details``
            from responses (OpenRouter extended thinking) and pass them
            back in subsequent messages so the model can continue reasoning.
        extra_body: Optional dict merged into every ``create()`` call's
            ``extra_body``.  Useful for OpenRouter-specific flags like
            ``{"reasoning": {"enabled": True}}``.
        extra_headers: Optional headers merged into every ``create()`` call.
            OpenRouter uses this for response caching and app attribution.
        cache_config: Optional default cache configuration. Supported keys:
            ``response_cache`` (bool), ``response_cache_ttl_seconds`` (int),
            ``response_cache_clear`` (bool), and ``cache_control``/``prompt_cache_control``
            (dict). Per-call ``cache_config`` values override these defaults.
    """

    def __init__(
        self,
        client: Any,
        *,
        preserve_reasoning: bool = True,
        extra_body: dict | None = None,
        extra_headers: dict | None = None,
        cache_config: dict | None = None,
    ) -> None:
        self._client = client
        self._preserve_reasoning = preserve_reasoning
        self._extra_body = extra_body or {}
        self._extra_headers = extra_headers or {}
        self._cache_config = cache_config or {}

    @property
    def client(self) -> Any:
        """The underlying OpenAI client."""
        return self._client

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        messages: list[Any],
        system_prompt: str | None,
        temperature: float,
        max_output_tokens: int,
        tool_schemas: Any | None = None,
        *,
        cache_config: dict | None = None,
    ) -> Any:
        # Build messages with system prompt prepended
        full_messages: list[dict] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        self._apply_request_options(kwargs, cache_config)

        return self._client.chat.completions.create(**kwargs)

    def generate_stream(
        self,
        model: str,
        messages: list[Any],
        system_prompt: str | None,
        temperature: float,
        max_output_tokens: int,
        tool_schemas: Any | None = None,
        *,
        cache_config: dict | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Any:
        full_messages: list[dict] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "stream": True,
        }
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        self._apply_request_options(kwargs, cache_config)

        text_chunks: list[str] = []
        usage = None
        role = "assistant"
        streamed_text = False
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        for chunk in self._client.chat.completions.create(**kwargs):
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            role = getattr(delta, "role", None) or role
            content = getattr(delta, "content", None)
            if content:
                text_chunks.append(content)
                if on_text_delta:
                    on_text_delta(content)
                    streamed_text = True

            for tc in getattr(delta, "tool_calls", None) or []:
                index = getattr(tc, "index", 0) or 0
                slot = tool_calls_by_index.setdefault(
                    index,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                if getattr(tc, "type", None):
                    slot["type"] = tc.type
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["function"]["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["function"]["arguments"] += fn.arguments

        tool_calls = [
            SimpleNamespace(
                id=(data.get("id") or f"call_{idx}"),
                type=data.get("type") or "function",
                function=SimpleNamespace(**data["function"]),
            )
            for idx, data in sorted(tool_calls_by_index.items())
        ]
        message = SimpleNamespace(
            role=role,
            content="".join(text_chunks) or None,
            tool_calls=tool_calls or None,
            reasoning_details=None,
            reasoning=None,
        )
        return _StreamedOpenAIResponse(
            message=message,
            usage=usage,
            streamed_text=streamed_text,
        )

    def parse_response(self, response: Any) -> ParsedResponse:
        choice = response.choices[0]
        message = choice.message

        text = getattr(message, "content", None)
        tool_calls: list[ToolCall] = []
        thinking_text: str | None = None

        message_tool_calls = getattr(message, "tool_calls", None)
        if message_tool_calls:
            for tc in message_tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=args,
                ))

        # OpenRouter reasoning support
        if self._preserve_reasoning:
            rd = getattr(message, "reasoning_details", None) or getattr(message, "reasoning", None)
            if rd:
                if isinstance(rd, list):
                    thinking_text = "".join(
                        item.get("content", "") if isinstance(item, dict) else str(item)
                        for item in rd
                    )
                elif isinstance(rd, str):
                    thinking_text = rd

        usage = TokenUsage()
        response_usage = getattr(response, "usage", None)
        if response_usage:
            usage.prompt_tokens = _read_field(response_usage, "prompt_tokens", 0) or 0
            usage.completion_tokens = _read_field(response_usage, "completion_tokens", 0) or 0
            details = _read_field(response_usage, "prompt_tokens_details", None)
            if details:
                usage.cached_tokens = _read_field(details, "cached_tokens", 0) or 0
                usage.cache_write_tokens = _read_field(details, "cache_write_tokens", 0) or 0

        raw_msg = self._message_to_dict(message)

        return ParsedResponse(
            text=text,
            tool_calls=tool_calls,
            raw_message=raw_msg,
            usage=usage,
            thinking_text=thinking_text,
            streamed_text=bool(getattr(response, "_agent_core_streamed_text", False)),
        )

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def build_user_message(self, text: str) -> Any:
        return {"role": "user", "content": text}

    def build_tool_result_messages(
        self,
        tool_calls: list[ToolCall],
        results: list[tuple[str, Any]],
    ) -> list[dict]:
        """Build one message per tool result (OpenAI requires explicit ID pairing)."""
        messages: list[dict] = []
        for tc, (_name, result) in zip(tool_calls, results):
            content = (
                result
                if isinstance(result, str)
                else json.dumps(_json_safe(result), ensure_ascii=True)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })
        return messages

    # ------------------------------------------------------------------
    # Tool schemas
    # ------------------------------------------------------------------

    def build_tool_schemas(self, callables: list[Callable]) -> Any | None:
        if not callables:
            return None
        from agent_core.providers._openai_schema import callable_to_openai_tool
        return [callable_to_openai_tool(f) for f in callables]

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(
        self,
        model: str,
        messages: list[Any],
        system_prompt: str | None = None,
    ) -> int:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            return 0

        total = 0
        if system_prompt:
            total += len(enc.encode(system_prompt)) + 4
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if content:
                    total += len(enc.encode(str(content))) + 4
                # Rough estimate for tool calls
                tc = msg.get("tool_calls")
                if tc:
                    total += len(enc.encode(json.dumps(tc))) + 4
            else:
                total += len(enc.encode(str(msg))) + 4
        return total

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def is_retryable_error(self, error: Exception) -> bool:
        try:
            import openai
            return isinstance(error, (openai.RateLimitError,))
        except ImportError:
            return False

    def get_retry_delay(
        self,
        error: Exception,
        attempt: int,
        base_delay: float,
        max_delay: float,
    ) -> float:
        # Check for Retry-After header
        headers = getattr(error, "headers", None) or {}
        retry_after = headers.get("retry-after") if isinstance(headers, dict) else None
        if retry_after:
            try:
                return min(float(retry_after), max_delay)
            except (ValueError, TypeError):
                pass
        return min(base_delay * (2 ** (attempt - 1)), max_delay)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_message(self, message: Any) -> dict:
        if isinstance(message, dict):
            return {**message, "_provider": "openai"}
        return {"_provider": "openai", "content": str(message)}

    def deserialize_message(self, data: dict) -> Any:
        d = dict(data)
        d.pop("_provider", None)
        return d

    def format_message_for_display(self, message: Any) -> dict | None:
        if not isinstance(message, dict):
            return {"role": "unknown", "content": str(message)}

        role = message.get("role", "unknown")
        parts: list[str] = []

        if message.get("content"):
            parts.append(str(message["content"]))

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                fn = tc.get("function", {})
                parts.append(
                    f"[Tool Call: {fn.get('name', '?')}({fn.get('arguments', '{}')})]"
                )

        if role == "tool":
            tool_id = message.get("tool_call_id", "?")
            parts = [f"[Tool Response: {tool_id} -> {message.get('content', '')}]"]

        return {"role": role, "content": "\n".join(parts)} if parts else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _message_to_dict(self, message: Any) -> dict:
        """Convert an OpenAI message object to a plain dict for history."""
        msg: dict[str, Any] = {"role": message.role}

        content = getattr(message, "content", None)
        if content:
            msg["content"] = content

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]

        # Preserve reasoning for OpenRouter
        if self._preserve_reasoning:
            rd = getattr(message, "reasoning_details", None) or getattr(message, "reasoning", None)
            if rd:
                msg["reasoning_details"] = rd

        return msg

    def _apply_request_options(
        self,
        kwargs: dict[str, Any],
        cache_config: dict | None,
    ) -> None:
        """Apply default extra options and per-call cache hints."""
        extra_body = dict(self._extra_body)
        extra_headers = dict(self._extra_headers)

        merged_cache = dict(self._cache_config)
        merged_cache.update(_normalize_cache_config(cache_config))

        prompt_cache_control = (
            merged_cache.get("prompt_cache_control")
            or merged_cache.get("cache_control")
        )
        if prompt_cache_control:
            extra_body["cache_control"] = prompt_cache_control

        response_cache = merged_cache.get("response_cache")
        if response_cache is not None:
            extra_headers["X-OpenRouter-Cache"] = (
                "true" if response_cache else "false"
            )

        response_cache_ttl = merged_cache.get("response_cache_ttl_seconds")
        if response_cache_ttl is not None:
            extra_headers["X-OpenRouter-Cache-TTL"] = str(response_cache_ttl)

        if merged_cache.get("response_cache_clear"):
            extra_headers["X-OpenRouter-Cache-Clear"] = "true"

        if extra_body:
            kwargs["extra_body"] = extra_body
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
