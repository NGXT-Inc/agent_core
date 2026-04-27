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
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlparse

from agent_core.core.attachments import format_attachment_placeholder
from agent_core.providers.types import (
    FilePart,
    FileOutputPart,
    ParsedResponse,
    ProviderCapabilities,
    TextPart,
    TextOutputPart,
    TokenUsage,
    ToolCall,
    UnsupportedInputPart,
    UserMessageInput,
    coerce_user_message,
)

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Return a JSON-compatible representation while preserving structure."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except TypeError:
            return _json_safe(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return _json_safe(value.dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, SimpleNamespace):
        return _json_safe(vars(value))
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _coerce_reasoning_details(value: Any) -> list[Any]:
    """Normalize streamed reasoning_details chunks into JSON-safe blocks."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return [_json_safe(value)]


def _reasoning_text(value: Any) -> str:
    """Extract displayable reasoning text from OpenRouter reasoning payloads."""
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(_reasoning_text(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("content", "text", "summary"):
            item = value.get(key)
            if isinstance(item, str) and item:
                parts.append(item)
        return "".join(parts)

    content = getattr(value, "content", None)
    text = getattr(value, "text", None)
    summary = getattr(value, "summary", None)
    return "".join(
        item for item in (content, text, summary) if isinstance(item, str) and item
    )


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either SDK objects or plain dicts."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _filename_from_uri(uri: str | None) -> str | None:
    if not uri:
        return None
    if uri.startswith("data:"):
        return None
    parsed = urlparse(uri)
    path = parsed.path or uri
    name = path.rsplit("/", 1)[-1]
    return name or None


def _mime_from_data_url(data_url: str | None) -> str | None:
    if not data_url or not data_url.startswith("data:"):
        return None
    header = data_url.split(",", 1)[0]
    mime = header[5:].split(";", 1)[0]
    return mime or None


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
        include_stream_usage: If ``True``, request usage accounting on streamed
            OpenAI-compatible responses where the API supports it.
        allow_file_urls: If ``True``, URI-backed non-image ``FilePart`` values
            are encoded as file content. Standard OpenAI Chat Completions does
            not support file URLs, but OpenRouter does.
        file_data_format: How inline non-image files are encoded in ``file``
            content parts: ``"base64"`` for OpenAI Chat, ``"data_url"`` for
            OpenRouter.
    """

    def __init__(
        self,
        client: Any,
        *,
        preserve_reasoning: bool = True,
        extra_body: dict | None = None,
        extra_headers: dict | None = None,
        cache_config: dict | None = None,
        include_stream_usage: bool = True,
        allow_file_urls: bool = False,
        file_data_format: str = "base64",
    ) -> None:
        self._client = client
        self._preserve_reasoning = preserve_reasoning
        self._extra_body = extra_body or {}
        self._extra_headers = extra_headers or {}
        self._cache_config = cache_config or {}
        self._include_stream_usage = include_stream_usage
        self._allow_file_urls = allow_file_urls
        if file_data_format not in {"base64", "data_url"}:
            raise ValueError("file_data_format must be 'base64' or 'data_url'")
        self._file_data_format = file_data_format

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
        if self._include_stream_usage:
            kwargs["stream_options"] = {"include_usage": True}
        self._apply_request_options(kwargs, cache_config)

        text_chunks: list[str] = []
        usage = None
        role = "assistant"
        streamed_text = False
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        reasoning_details: list[Any] = []
        reasoning_chunks: list[str] = []

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

            delta_reasoning_details = getattr(delta, "reasoning_details", None)
            if delta_reasoning_details:
                reasoning_details.extend(
                    _coerce_reasoning_details(delta_reasoning_details)
                )

            delta_reasoning = (
                getattr(delta, "reasoning", None)
                or getattr(delta, "reasoning_content", None)
            )
            if delta_reasoning:
                reasoning_chunks.append(str(delta_reasoning))

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
            reasoning_details=reasoning_details or None,
            reasoning="".join(reasoning_chunks) or None,
        )
        return _StreamedOpenAIResponse(
            message=message,
            usage=usage,
            streamed_text=streamed_text,
        )

    def parse_response(self, response: Any) -> ParsedResponse:
        choice = response.choices[0]
        message = choice.message

        text = self._content_to_display_text(getattr(message, "content", None) or "")
        tool_calls: list[ToolCall] = []
        thinking_text: str | None = None
        output_parts: list[TextOutputPart | FileOutputPart] = []
        if text:
            output_parts.append(TextOutputPart(text))

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
                thinking_text = _reasoning_text(rd)

        usage = TokenUsage()
        response_usage = getattr(response, "usage", None)
        if response_usage:
            usage.prompt_tokens = _read_field(response_usage, "prompt_tokens", 0) or 0
            usage.completion_tokens = _read_field(response_usage, "completion_tokens", 0) or 0
            details = _read_field(response_usage, "prompt_tokens_details", None)
            if details:
                usage.cached_tokens = _read_field(details, "cached_tokens", 0) or 0
                usage.cache_write_tokens = _read_field(details, "cache_write_tokens", 0) or 0

        for image in getattr(message, "images", None) or []:
            file_output = self._file_output_from_image_output(image)
            if file_output is not None:
                output_parts.append(file_output)

        raw_msg = self._message_to_dict(message)

        return ParsedResponse(
            text=text,
            tool_calls=tool_calls,
            raw_message=raw_msg,
            usage=usage,
            thinking_text=thinking_text,
            streamed_text=bool(getattr(response, "_agent_core_streamed_text", False)),
            output_parts=[] if tool_calls else output_parts,
        )

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_file_content_part(self, part: FilePart) -> dict[str, Any]:
        if part.is_image and part.file_id is None:
            if part.uri is not None:
                image_url = part.uri
            elif part.data is not None:
                image_url = part.to_data_url()
            else:
                raise UnsupportedInputPart(
                    "OpenAIProvider does not support image file IDs through "
                    "Chat Completions"
                )
            payload: dict[str, Any] = {"url": image_url}
            if part.detail:
                payload["detail"] = part.detail
            return {"type": "image_url", "image_url": payload}

        file_payload: dict[str, Any] = {}
        if part.filename:
            file_payload["filename"] = part.filename

        if part.file_id is not None:
            file_payload["file_id"] = part.file_id
        elif part.data is not None:
            file_payload["file_data"] = (
                part.to_data_url()
                if self._file_data_format == "data_url"
                else part.to_base64()
            )
        elif part.uri is not None:
            if not self._allow_file_urls:
                raise UnsupportedInputPart(
                    "OpenAIProvider does not support non-image file URI inputs "
                    "through Chat Completions; use inline data, file_id, or a "
                    "Responses provider."
                )
            file_payload["file_data"] = part.uri

        return {"type": "file", "file": file_payload}

    def build_user_message(self, message: UserMessageInput) -> Any:
        user_message = coerce_user_message(message)
        if all(isinstance(part, TextPart) for part in user_message.parts):
            return {
                "role": "user",
                "content": "\n".join(
                    part.text
                    for part in user_message.parts
                    if isinstance(part, TextPart)
                ),
            }

        content_parts: list[dict[str, Any]] = []
        for part in user_message.parts:
            if isinstance(part, TextPart):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, FilePart):
                content_parts.append({"type": "text", "text": part.live_label()})
                content_parts.append(self._build_file_content_part(part))
        return {
            "role": "user",
            "content": content_parts,
        }

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
                content = self._content_to_display_text(msg.get("content", ""))
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
    # Runtime capabilities
    # ------------------------------------------------------------------

    def supports_context_cache_registry(self, cache_registry: Any) -> bool:
        """OpenAI-compatible providers do not use Gemini explicit caches."""
        return False

    def capabilities(self, model: str | None = None) -> ProviderCapabilities:
        """Return OpenAI-compatible chat-completions input/output support."""
        return ProviderCapabilities(
            input_images=True,
            input_image_bytes=True,
            input_image_urls=True,
            input_files=True,
            input_file_bytes=True,
            input_file_urls=self._allow_file_urls,
            input_file_ids=True,
            output_files=True,
            tool_calling=True,
            streaming=True,
        )

    def adjust_compaction_tail_start(self, messages: list[Any], start: int) -> int:
        """Keep OpenAI tool-result messages paired with assistant tool calls."""
        start = max(0, min(start, len(messages)))
        while (
            start > 0
            and start < len(messages)
            and self._is_tool_result_message(messages[start])
        ):
            start -= 1
        return start

    @staticmethod
    def _is_tool_result_message(message: Any) -> bool:
        return isinstance(message, dict) and message.get("role") == "tool"

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
            serialized = dict(message)
            if "images" in serialized:
                image_placeholders = [
                    placeholder
                    for image in serialized.pop("images") or []
                    if (
                        placeholder := self._placeholder_from_image_output(image)
                    ) is not None
                ]
                serialized["content"] = self._append_placeholders_to_content(
                    serialized.get("content"),
                    image_placeholders,
                )
            if "content" in serialized:
                serialized["content"] = self._sanitize_content_for_persistence(
                    serialized["content"]
                )
            return {**serialized, "_provider": "openai"}
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
            parts.append(self._content_to_display_text(message["content"]))

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

    def _placeholder_from_content_part(self, part: dict[str, Any]) -> str:
        part_type = part.get("type", "file")

        if part_type == "image_url":
            image_url = part.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            mime_type = _mime_from_data_url(url) or "image/*"
            return format_attachment_placeholder(
                mime_type=mime_type,
                label=_filename_from_uri(url) or "image attachment",
            )

        if part_type == "file":
            file_payload = part.get("file") or {}
            if isinstance(file_payload, dict):
                file_data = file_payload.get("file_data") or file_payload.get("fileData")
                mime_type = _mime_from_data_url(file_data) or "application/octet-stream"
                label = (
                    file_payload.get("filename")
                    or _filename_from_uri(file_data)
                    or file_payload.get("file_id")
                )
            else:
                mime_type = "application/octet-stream"
                label = None
            return format_attachment_placeholder(mime_type=mime_type, label=label)

        if part_type == "input_audio":
            return format_attachment_placeholder(
                mime_type="audio/*",
                label="audio attachment",
            )

        return format_attachment_placeholder(
            mime_type="application/octet-stream",
            label=f"{part_type} attachment",
        )

    def _file_output_from_image_output(self, image: Any) -> FileOutputPart | None:
        image_url = _read_field(image, "image_url", None) or {}
        url = _read_field(image_url, "url", None)
        if not url:
            return None

        filename = _filename_from_uri(url)
        if isinstance(url, str) and url.startswith("data:"):
            return FileOutputPart.from_data_url(url, filename=filename)
        return FileOutputPart(
            uri=url,
            mime_type=_mime_from_data_url(url) or "image/*",
            filename=filename,
        )

    def _placeholder_from_image_output(self, image: Any) -> str | None:
        image_url = _read_field(image, "image_url", None) or {}
        url = _read_field(image_url, "url", None)
        if not url:
            return None
        return self._placeholder_from_content_part({
            "type": "image_url",
            "image_url": {"url": url},
        })

    def _append_placeholders_to_content(
        self,
        content: Any,
        placeholders: list[str],
    ) -> Any:
        if not placeholders:
            return content

        content_text = self._content_to_display_text(content) if content else ""
        pieces = [piece for piece in [content_text, *placeholders] if piece]
        return "\n".join(pieces)

    def _sanitize_content_for_persistence(self, content: Any) -> Any:
        if not isinstance(content, list):
            return content

        sanitized: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                sanitized.append({"type": "text", "text": str(part)})
                continue
            if part.get("type") == "text":
                sanitized.append(dict(part))
            else:
                sanitized.append({
                    "type": "text",
                    "text": self._placeholder_from_content_part(part),
                })
        return sanitized

    def _content_to_display_text(self, content: Any) -> str:
        if isinstance(content, list):
            pieces: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    pieces.append(str(part))
                elif part.get("type") == "text":
                    pieces.append(str(part.get("text", "")))
                else:
                    pieces.append(self._placeholder_from_content_part(part))
            return "\n".join(piece for piece in pieces if piece)
        return str(content)

    def _message_to_dict(self, message: Any) -> dict:
        """Convert an OpenAI message object to a plain dict for history."""
        msg: dict[str, Any] = {"role": message.role}

        content = getattr(message, "content", None)
        image_placeholders = [
            placeholder
            for image in getattr(message, "images", None) or []
            if (placeholder := self._placeholder_from_image_output(image)) is not None
        ]
        content = self._append_placeholders_to_content(content, image_placeholders)
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
                if getattr(message, "reasoning_details", None):
                    msg["reasoning_details"] = _json_safe(rd)
                else:
                    msg["reasoning"] = _reasoning_text(rd) or str(rd)

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
        if merged_cache.get("response_cache_clear"):
            response_cache = True
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
