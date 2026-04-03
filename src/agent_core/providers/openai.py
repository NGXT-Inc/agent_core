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
from typing import Any, Callable

from agent_core.providers.types import ParsedResponse, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        client: Any,
        *,
        preserve_reasoning: bool = True,
        extra_body: dict | None = None,
    ) -> None:
        self._client = client
        self._preserve_reasoning = preserve_reasoning
        self._extra_body = extra_body or {}

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
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        return self._client.chat.completions.create(**kwargs)

    def parse_response(self, response: Any) -> ParsedResponse:
        choice = response.choices[0]
        message = choice.message

        text = message.content
        tool_calls: list[ToolCall] = []
        thinking_text: str | None = None

        if message.tool_calls:
            for tc in message.tool_calls:
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
        if response.usage:
            usage.prompt_tokens = response.usage.prompt_tokens or 0
            usage.completion_tokens = response.usage.completion_tokens or 0

        raw_msg = self._message_to_dict(message)

        return ParsedResponse(
            text=text,
            tool_calls=tool_calls,
            raw_message=raw_msg,
            usage=usage,
            thinking_text=thinking_text,
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
            content = json.dumps(result) if isinstance(result, dict) else str(result)
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

        if message.content:
            msg["content"] = message.content

        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        # Preserve reasoning for OpenRouter
        if self._preserve_reasoning:
            rd = getattr(message, "reasoning_details", None) or getattr(message, "reasoning", None)
            if rd:
                msg["reasoning_details"] = rd

        return msg
