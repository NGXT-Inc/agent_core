"""Provider-neutral types for multi-LLM support.

These types form the boundary between the provider-agnostic orchestration
loop in Agent and the provider-specific implementations (Gemini, OpenAI, etc.).

Messages in history remain provider-specific (opaque ``Any``). The loop reads
responses through ``ParsedResponse``, which is the only neutral type it
inspects during execution.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, TypeAlias, runtime_checkable

from agent_core.core.attachments import format_attachment_placeholder


class UnsupportedInputPart(ValueError):
    """Raised when a provider cannot encode a user message part."""


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Provider/model features used for early input validation.

    Empty ``supported_input_mime_types`` means the provider accepts any MIME
    type for the modalities and sources it otherwise supports. Entries may be
    exact MIME types (``"application/pdf"``) or type wildcards
    (``"image/*"``).
    """

    input_text: bool = True
    input_images: bool = False
    input_image_bytes: bool = False
    input_image_urls: bool = False
    input_image_file_ids: bool = False
    input_files: bool = False
    input_file_bytes: bool = False
    input_file_urls: bool = False
    input_file_ids: bool = False
    output_text: bool = True
    output_files: bool = False
    tool_calling: bool = True
    streaming: bool = True
    supported_input_mime_types: tuple[str, ...] = ()

    def supports_input_mime_type(self, mime_type: str) -> bool:
        """Return whether *mime_type* is allowed by the provider."""
        if not self.supported_input_mime_types:
            return True

        normalized = (mime_type or "application/octet-stream").lower()
        for allowed in self.supported_input_mime_types:
            candidate = allowed.lower()
            if candidate in {"*", "*/*"}:
                return True
            if candidate.endswith("/*") and normalized.startswith(candidate[:-1]):
                return True
            if candidate == normalized:
                return True
        return False


@dataclass(frozen=True, slots=True)
class TextPart:
    """A text segment inside a provider-neutral user message."""

    text: str


@dataclass(frozen=True, slots=True)
class FilePart:
    """A provider-neutral file attachment.

    Attachments are ephemeral by default. Live provider messages may contain
    bytes, URLs, or provider file IDs, but persistence should retain only a
    text placeholder unless an application supplies its own durable attachment
    store.
    """

    data: bytes | None = field(default=None, repr=False)
    uri: str | None = None
    file_id: str | None = None
    mime_type: str = "application/octet-stream"
    filename: str | None = None
    description: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        sources = [
            self.data is not None,
            self.uri is not None,
            self.file_id is not None,
        ]
        if sum(sources) != 1:
            raise ValueError(
                "FilePart requires exactly one of data, uri, or file_id"
            )
        if not self.mime_type:
            object.__setattr__(self, "mime_type", "application/octet-stream")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
        description: str | None = None,
        detail: str | None = None,
    ) -> "FilePart":
        """Create an inline attachment from a local file path."""
        file_path = Path(path)
        guessed_type, _encoding = mimetypes.guess_type(file_path.name)
        return cls(
            data=file_path.read_bytes(),
            mime_type=mime_type or guessed_type or "application/octet-stream",
            filename=filename or file_path.name,
            description=description,
            detail=detail,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        mime_type: str,
        filename: str | None = None,
        description: str | None = None,
        detail: str | None = None,
    ) -> "FilePart":
        return cls(
            data=data,
            mime_type=mime_type,
            filename=filename,
            description=description,
            detail=detail,
        )

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        mime_type: str,
        filename: str | None = None,
        description: str | None = None,
        detail: str | None = None,
    ) -> "FilePart":
        return cls(
            uri=uri,
            mime_type=mime_type,
            filename=filename,
            description=description,
            detail=detail,
        )

    @classmethod
    def from_file_id(
        cls,
        file_id: str,
        *,
        mime_type: str = "application/octet-stream",
        filename: str | None = None,
        description: str | None = None,
        detail: str | None = None,
    ) -> "FilePart":
        return cls(
            file_id=file_id,
            mime_type=mime_type,
            filename=filename,
            description=description,
            detail=detail,
        )

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")

    def to_data_url(self) -> str:
        """Return inline bytes as a data URL for OpenAI-compatible providers."""
        if self.data is None:
            raise ValueError("Only inline FilePart data can be encoded as a data URL")
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"

    def to_base64(self) -> str:
        """Return inline bytes as plain base64."""
        if self.data is None:
            raise ValueError("Only inline FilePart data can be base64 encoded")
        return base64.b64encode(self.data).decode("ascii")

    def placeholder(self) -> str:
        """Describe an ephemeral attachment for persisted/reloaded history."""
        label = self.filename or self.description or "unnamed attachment"
        return format_attachment_placeholder(mime_type=self.mime_type, label=label)

    def live_label(self) -> str:
        """Describe an attachment to the live model before the binary/URI part."""
        label = self.filename or self.description or "unnamed attachment"
        return f"[Attached file: {label} ({self.mime_type})]"


MessagePart: TypeAlias = TextPart | FilePart


@dataclass(frozen=True, slots=True)
class TextOutputPart:
    """A text segment returned by a model."""

    text: str


@dataclass(frozen=True, slots=True)
class FileOutputPart:
    """A file-like output returned by a model."""

    data: bytes | None = field(default=None, repr=False)
    uri: str | None = None
    mime_type: str = "application/octet-stream"
    filename: str | None = None

    def __post_init__(self) -> None:
        if (self.data is None) == (self.uri is None):
            raise ValueError("FileOutputPart requires exactly one of data or uri")
        if not self.mime_type:
            object.__setattr__(self, "mime_type", "application/octet-stream")

    @classmethod
    def from_data_url(
        cls,
        data_url: str,
        *,
        filename: str | None = None,
    ) -> "FileOutputPart":
        if not data_url.startswith("data:") or "," not in data_url:
            raise ValueError("Expected a data URL")
        header, encoded = data_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
        return cls(
            data=base64.b64decode(encoded),
            mime_type=mime_type,
            filename=filename,
        )


OutputPart: TypeAlias = TextOutputPart | FileOutputPart


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Structured agent response with a text convenience view."""

    parts: tuple[OutputPart, ...]
    token_usage: dict | None = None

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        token_usage: dict | None = None,
    ) -> "AgentResponse":
        return cls(parts=(TextOutputPart(text),), token_usage=token_usage)

    @property
    def text(self) -> str:
        return "\n".join(
            part.text for part in self.parts if isinstance(part, TextOutputPart)
        )

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class UserMessage:
    """Provider-neutral user message made of text and attachment parts."""

    parts: tuple[MessagePart, ...]

    def __init__(self, parts: list[MessagePart] | tuple[MessagePart, ...]):
        if not parts:
            raise ValueError("UserMessage requires at least one part")
        object.__setattr__(self, "parts", tuple(parts))

    @classmethod
    def from_text(cls, text: str) -> "UserMessage":
        return cls([TextPart(text)])

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        attachments: list[FilePart] | tuple[FilePart, ...] | None = None,
    ) -> "UserMessage":
        parts: list[MessagePart] = [TextPart(prompt)]
        if attachments:
            parts.extend(attachments)
        return cls(parts)

    @property
    def text(self) -> str:
        """Text-only view used for hooks, events, and display."""
        chunks: list[str] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                chunks.append(part.text)
            elif isinstance(part, FilePart):
                chunks.append(part.placeholder())
        return "\n".join(chunk for chunk in chunks if chunk)


UserMessageInput: TypeAlias = str | UserMessage


def coerce_user_message(
    message: UserMessageInput,
    attachments: list[FilePart] | tuple[FilePart, ...] | None = None,
) -> UserMessage:
    """Normalize public user-message inputs to ``UserMessage``."""
    if isinstance(message, UserMessage):
        if attachments:
            return UserMessage([*message.parts, *attachments])
        return message
    return UserMessage.from_prompt(message, attachments)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation extracted from a model response.

    Provider-neutral: Gemini generates a UUID for ``id`` (since it has no
    explicit call IDs), OpenAI uses the ``id`` from the response.
    """

    id: str
    name: str
    args: dict


@dataclass(slots=True)
class TokenUsage:
    """Accumulated token counts across a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(slots=True)
class ParsedResponse:
    """Provider-neutral view of a model response.

    The orchestration loop uses ``text``, ``tool_calls``, and ``usage``
    to drive its logic.  ``raw_message`` is an opaque provider-specific
    object that gets appended to the conversation history as-is.
    """

    text: str | None
    tool_calls: list[ToolCall]
    raw_message: Any
    usage: TokenUsage = field(default_factory=TokenUsage)
    thinking_text: str | None = None
    streamed_text: bool = False
    output_parts: list[OutputPart] = field(default_factory=list)
    finish_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that every LLM backend must implement.

    Methods fall into five groups:

    1. **Generation** — call the model and parse its response.
    2. **Message construction** — build provider-specific messages.
    3. **Runtime capabilities** — advertise cache and compaction behavior.
    4. **Error handling** — classify exceptions for the retry loop.
    5. **Serialization** — persist / display messages.
    """

    # --- Generation ---

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
        """Send messages to the model and return the raw response.

        Args:
            model: Model identifier (e.g. ``"gemini-3.1-pro-preview"``).
            messages: Full conversation history in provider-specific format.
            system_prompt: System instruction text (may be ``None``).
            temperature: Sampling temperature.
            max_output_tokens: Maximum tokens in the response.
            tool_schemas: Provider-specific tool declarations, or ``None``
                if the agent has no tools.
            cache_config: Optional provider-specific caching hints.
                Gemini uses ``{"cache_name": str, "contents_offset": int}``.
                Providers that don't support caching ignore this.
        """
        ...

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
        """Stream a model response and return the final aggregated response.

        Implementations should call *on_text_delta* for text chunks as they
        arrive, but still return a complete provider-specific response that
        ``parse_response()`` can consume normally. Providers that cannot stream
        may delegate to ``generate()`` and optionally emit the final text once.
        """
        ...

    def parse_response(self, response: Any) -> ParsedResponse:
        """Extract text, tool calls, and usage from a raw response."""
        ...

    # --- Message construction ---

    def build_user_message(self, message: UserMessageInput) -> Any:
        """Create a user message in provider-specific format."""
        ...

    def build_tool_result_messages(
        self,
        tool_calls: list[ToolCall],
        results: list[tuple[str, Any]],
    ) -> Any:
        """Build message(s) containing tool execution results.

        Returns either a single message object or a list of messages.
        Gemini returns one ``Content`` with multiple function-response Parts.
        OpenAI returns a list of ``{"role": "tool", ...}`` dicts (one per tool).
        """
        ...

    def build_tool_schemas(self, callables: list[Callable]) -> Any | None:
        """Convert Python callables to provider-specific tool declarations.

        Returns ``None`` when *callables* is empty.
        """
        ...

    # --- Token counting ---

    def count_tokens(
        self,
        model: str,
        messages: list[Any],
        system_prompt: str | None = None,
    ) -> int:
        """Count tokens in the current context.

        Best-effort — providers without a token-counting API may return 0.
        """
        ...

    # --- Runtime capabilities ---

    def supports_context_cache_registry(self, cache_registry: Any) -> bool:
        """Return whether this provider can use the given explicit cache registry.

        The built-in ``ContextCacheRegistry`` creates Vertex/Gemini cached
        content. OpenAI-compatible providers use request-level cache controls
        instead and should return ``False``.
        """
        ...

    def adjust_compaction_tail_start(self, messages: list[Any], start: int) -> int:
        """Move a proposed compaction tail start to a provider-valid boundary.

        Providers with paired tool-call/tool-result messages should move
        ``start`` backward when needed so the preserved tail remains a valid
        conversation transcript.
        """
        ...

    def capabilities(self, model: str | None = None) -> ProviderCapabilities:
        """Return provider/model feature support for validation and routing.

        Providers may ignore *model* when capabilities are not model-specific.
        """
        ...

    # --- Error handling ---

    def is_retryable_error(self, error: Exception) -> bool:
        """Return ``True`` if the error is transient and retryable (e.g. 429)."""
        ...

    def get_retry_delay(
        self,
        error: Exception,
        attempt: int,
        base_delay: float,
        max_delay: float,
    ) -> float:
        """Compute retry delay in seconds for a retryable error."""
        ...

    # --- Serialization ---

    def serialize_message(self, message: Any) -> dict:
        """Serialize a provider message to a JSON-safe dict."""
        ...

    def deserialize_message(self, data: dict) -> Any:
        """Deserialize a JSON dict back to a provider message."""
        ...

    def format_message_for_display(self, message: Any) -> dict | None:
        """Format a message for human-readable display (e.g. ``get_context()``).

        Returns ``{"role": str, "content": str}`` or ``None`` to skip.
        """
        ...
