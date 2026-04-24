"""Provider-neutral types for multi-LLM support.

These types form the boundary between the provider-agnostic orchestration
loop in Agent and the provider-specific implementations (Gemini, OpenAI, etc.).

Messages in history remain provider-specific (opaque ``Any``). The loop reads
responses through ``ParsedResponse``, which is the only neutral type it
inspects during execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


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


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that every LLM backend must implement.

    Methods fall into four groups:

    1. **Generation** — call the model and parse its response.
    2. **Message construction** — build provider-specific messages.
    3. **Error handling** — classify exceptions for the retry loop.
    4. **Serialization** — persist / display messages.
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

    def build_user_message(self, text: str) -> Any:
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
