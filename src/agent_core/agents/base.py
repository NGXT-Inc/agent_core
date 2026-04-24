"""Base agent class with LLM function-calling support.

Each agent is an LLM instance with:
- A specific system prompt defining its role
- A set of tools (Python functions) it can call
- Manual function calling loop with parallel execution support

Supports multiple LLM backends via the LLMProvider protocol.
Default backend is Gemini (Vertex AI) for backward compatibility.

This module is designed to be extended without modification:
- Override class attributes for configuration
- Override hook methods for custom behavior
- Inject custom ConversationStore for persistence
- Inject a custom LLMProvider for non-Gemini models
"""

import functools
import logging
import os
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, ClassVar

logger = logging.getLogger(__name__)

from dotenv import find_dotenv, load_dotenv
from google import genai

from agent_core.agents.compaction import (
    CompactionConfig,
    approximate_tokens,
    estimate_history_tokens,
    render_message_for_compaction,
    select_preserved_tail_start,
    trimmed_transcript_lines,
)
from agent_core.core.caching import ContextCacheRegistry
from agent_core.core.events import EventBus, EventType, EventStatus, Event, get_event_bus, emit_event
from agent_core.core.persistence import ConversationStoreProtocol
from agent_core.providers.types import LLMProvider, ParsedResponse, TokenUsage, ToolCall

# Default model constants (can be overridden at class level)
MODEL_PRO = "gemini-3.1-pro-preview"
MODEL_FLASH = "gemini-3-flash-preview"
MODEL_FLASH_LITE = "gemini-3.1-flash-lite-preview"


_env_loaded = False


def _ensure_env() -> None:
    """Load .env file once, lazily."""
    global _env_loaded
    if not _env_loaded:
        load_dotenv(find_dotenv(usecwd=True))
        _env_loaded = True


def _resolve_env() -> tuple[str | None, str]:
    """Resolve Google Cloud env vars, loading .env if present.

    Called lazily on first Agent that needs auto-client creation,
    not at import time.
    """
    _ensure_env()
    project = os.environ.get("GOOGLE_PROJECT_ID")
    location = os.environ.get("GOOGLE_LOCATION", "global")
    return project, location


# Backward-compatible lazy accessors. Downstream code does:
#   from agent_core.agents.base import GOOGLE_PROJECT_ID, GOOGLE_LOCATION
# and uses them inside functions (e.g., project=GOOGLE_PROJECT_ID).
# Module __getattr__ resolves them to real strings on first access.
_LAZY_ENV_VARS = {
    "GOOGLE_PROJECT_ID": ("GOOGLE_PROJECT_ID", None),
    "GOOGLE_LOCATION": ("GOOGLE_LOCATION", "global"),
}


def __getattr__(name: str):
    if name in _LAZY_ENV_VARS:
        env_key, default = _LAZY_ENV_VARS[name]
        _ensure_env()
        return os.environ.get(env_key, default)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate_instance_id(
    agent_type: str,
    session_id: str | None = None,
    root_agent_types: set[str] | None = None,
) -> str:
    """Generate a unique instance ID for an agent invocation.

    Args:
        agent_type: The type of agent (e.g., "designer", "implementer").
        session_id: Optional session ID. If provided and agent is a root type,
                   uses first 8 chars for deterministic ID.
        root_agent_types: Set of agent types considered "root" agents.
                         Defaults to Agent.ROOT_AGENT_TYPES.

    Returns:
        Unique ID like "designer_abc123".
    """
    if root_agent_types is None:
        root_agent_types = Agent.ROOT_AGENT_TYPES

    if session_id and agent_type in root_agent_types:
        # Use session-based ID for deterministic identification
        # This allows frontend and backend to generate matching IDs
        short_id = session_id[:8]
    else:
        # Use random UUID for sub-agents (multiple instances per session)
        short_id = uuid.uuid4().hex[:8]
    return f"{agent_type}_{short_id}"


class Agent:
    """Base class for all agents.

    Agents are provider-backed LLM runtimes with specific prompts and tools.
    Uses manual function calling loop with parallel execution support.

    Class Attributes (override in subclasses):
        name: Agent type identifier.
        system_prompt: Default system prompt.
        DEFAULT_MODEL: Model to use if not specified.
        ROOT_AGENT_TYPES: Agent types that get deterministic IDs from session.
        CODE_TOOLS: Tool names that handle code (for special result formatting).
        MAX_PARALLEL_TOOLS: Maximum concurrent tool executions.
        MAX_ITERATIONS: Maximum function calling loop iterations.
        emit_lifecycle_events: Whether to emit AGENT_START/END events.
        emit_tool_events: Whether to emit TOOL_START/END events.

    Usage:
        class ResearcherAgent(Agent):
            name = "researcher"
            system_prompt = "You are a literature research assistant..."

            def __init__(self, session_id: str = None):
                super().__init__(session_id=session_id)
                self.register_tool(self.search_papers)
                self.register_tool(self.read_paper)

            def search_papers(self, query: str) -> list[dict]:
                '''Search for research papers.'''
                ...
    """

    # --- Class-level configuration (override in subclasses) ---

    name: str = "base"
    system_prompt: str = "You are a helpful assistant."

    # Default model (override per agent type)
    DEFAULT_MODEL: ClassVar[str] = MODEL_PRO

    # Agent types that get deterministic instance IDs from session_id.
    # Override in subclasses with a frozenset that includes your agent's name:
    #   ROOT_AGENT_TYPES = frozenset({"my_agent", *Agent.ROOT_AGENT_TYPES})
    ROOT_AGENT_TYPES: ClassVar[frozenset[str]] = frozenset({"designer", "analyst", "data_analyst"})

    # Tools that handle code execution (for special result formatting in events)
    # Override in subclasses: e.g. CODE_TOOLS = {"execute_code"}
    CODE_TOOLS: ClassVar[set[str]] = set()

    # Execution limits
    MAX_PARALLEL_TOOLS: ClassVar[int] = 10
    MAX_ITERATIONS: ClassVar[int] = 50

    # Context caching
    ENABLE_CACHING: ClassVar[bool] = True
    CACHE_MIN_TOKENS: ClassVar[int] = 32_768

    # Context compaction
    ENABLE_COMPACTION: ClassVar[bool] = False

    # Streaming
    DEFAULT_STREAMING: ClassVar[bool] = False

    # Shared cache registry (initialized once at app startup)
    _cache_registry: ClassVar[ContextCacheRegistry | None] = None

    # Event emission flags (set False to handle events yourself)
    emit_lifecycle_events: ClassVar[bool] = True
    emit_tool_events: ClassVar[bool] = True

    @classmethod
    def init_cache_registry(
        cls,
        client: genai.Client,
        max_workers: int = 4,
        cache_ttl_seconds: int = 600,
    ) -> None:
        """Initialize the shared cache registry. Call once at app startup.

        .. deprecated::
            Prefer passing ``cache_registry`` to ``Agent.__init__()``
            for per-agent-tree isolation.
        """
        if cls._cache_registry is not None:
            cls._cache_registry.close()
        cls._cache_registry = ContextCacheRegistry(
            client, max_workers, cache_ttl_seconds
        )

    @classmethod
    def shutdown_cache_registry(cls) -> None:
        """Shutdown the shared cache registry. Call once at app teardown.

        .. deprecated::
            Prefer passing ``cache_registry`` to ``Agent.__init__()``
            and managing its lifecycle directly.
        """
        if cls._cache_registry:
            cls._cache_registry.close()
            cls._cache_registry = None

    def __init__(
        self,
        model_name: str | None = None,
        client: genai.Client | None = None,
        provider: LLMProvider | None = None,
        parent_agent: str | None = None,
        session_id: str | None = None,
        conversation_store: ConversationStoreProtocol | None = None,
        cancel_event: threading.Event | None = None,
        event_bus: EventBus | None = None,
        cache_registry: ContextCacheRegistry | None = None,
        streaming: bool | None = None,
    ):
        """Initialize the agent.

        Args:
            model_name: Model to use. Defaults to class DEFAULT_MODEL.
            client: Optional pre-configured Gemini client. Wraps it in a
                   GeminiProvider. Ignored if *provider* is given.
            provider: Optional LLMProvider instance. Takes priority over
                     *client*. If neither is given, a GeminiProvider is
                     created from environment variables.
            parent_agent: Instance ID of the parent agent (for graph visualization).
            session_id: Optional session ID for deterministic instance ID generation
                       and conversation persistence.
            conversation_store: Optional persistence backend. If None, history is
                              kept in memory for the lifetime of this agent.
            cancel_event: Optional shared threading.Event for cancellation.
                         If provided, this agent shares the cancel signal with
                         its parent — calling cancel() on either stops both.
                         If None, the agent creates its own independent event.
            event_bus: Optional EventBus instance. If None, uses the global
                      singleton from get_event_bus(). Pass a custom instance
                      to isolate event streams (e.g., in tests).
            cache_registry: Optional ContextCacheRegistry instance. If None,
                          falls back to the class-level _cache_registry.
                          Pass a custom instance for per-agent-tree isolation.
            streaming: Optional default for run()/run_stateless(). If None,
                       uses class DEFAULT_STREAMING. Per-call arguments
                       still take priority.
        """
        from agent_core.providers.gemini import GeminiProvider

        # Provider resolution: provider > client > auto-create from env
        if provider is not None:
            self._provider = provider
        elif client is not None:
            self._provider = GeminiProvider(client=client)
        else:
            project_id, location = _resolve_env()
            if not project_id:
                raise ValueError("GOOGLE_PROJECT_ID must be set in environment or .env file")
            self._provider = GeminiProvider(
                client=genai.Client(
                    vertexai=True, project=project_id, location=location
                )
            )

        self.model_name = model_name or self.DEFAULT_MODEL
        self.streaming = self.DEFAULT_STREAMING if streaming is None else streaming

        # Generate unique instance ID
        self.instance_id = generate_instance_id(self.name, session_id, self.ROOT_AGENT_TYPES)

        # Store session_id for persistence
        self._session_id: str | None = session_id

        # Tools registry - maps function name to callable
        self._tools: dict[str, Callable] = {}

        # Raw callables (for cache fingerprinting and cache creation conversion)
        self._tool_functions: list[Callable] = []

        # Provider-specific tool declarations (built via provider.build_tool_schemas)
        self._tool_schemas: Any | None = None

        # Conversation persistence
        self._conversation_store = conversation_store
        if session_id and self._conversation_store:
            self._history: list[Any] = self._conversation_store.load(
                session_id, self.name
            )
        else:
            self._history: list[Any] = []
        self._compaction_count = 0

        # Parent agent instance ID for event tracking (graph edges)
        self._parent_agent: str | None = parent_agent

        # Wave ID for grouping parallel agents in visualization
        self._wave_id: str | None = None

        # Cancellation support (thread-safe, shareable across agent tree)
        self._owns_cancel_event = cancel_event is None
        self._cancel_event = cancel_event if cancel_event is not None else threading.Event()

        # Event bus (instance-level, falls back to global singleton)
        self._event_bus = event_bus if event_bus is not None else get_event_bus()

        # Cache registry (instance-level, falls back to class-level default)
        self._instance_cache_registry = cache_registry if cache_registry is not None else self._cache_registry

        # Register with cache registry (only for persistent sessions —
        # avoids overhead on short-lived sub-agents using run_stateless)
        if (
            self.ENABLE_CACHING
            and session_id
            and self._instance_cache_registry
            and self._provider_supports_context_cache_registry()
        ):
            self._instance_cache_registry.register(
                agent_id=self.instance_id,
                model_name=self.model_name,
                min_token_threshold=self.CACHE_MIN_TOKENS,
            )
            self._cache_enabled = True
        else:
            self._cache_enabled = False

    def _provider_supports_context_cache_registry(self) -> bool:
        """Return whether the active provider can use the configured registry."""
        supports = getattr(self._provider, "supports_context_cache_registry", None)
        if supports is None:
            return False
        try:
            return bool(supports(self._instance_cache_registry))
        except Exception as exc:
            logger.warning(
                "[%s] disabling context cache: provider capability check failed: %s",
                self.name,
                exc,
            )
            return False

    # --- Lifecycle Hooks (override in subclasses) ---

    def on_agent_start(self, prompt: str) -> None:
        """Called when agent starts processing a prompt.

        Override to add custom logging, metrics, or setup.

        Args:
            prompt: The user prompt being processed.
        """
        pass

    def on_agent_end(
        self,
        result: str,
        success: bool,
        error: str | None = None,
        cancelled: bool = False,
    ) -> None:
        """Called when agent finishes processing.

        Override to add custom logging, cleanup, or metrics.

        Args:
            result: The final result text (empty if failed).
            success: Whether processing completed successfully.
            error: Error message if failed.
            cancelled: Whether the run was cancelled via cancel().
        """
        pass

    def on_tool_start(self, tool_name: str, args: dict, tool_call_id: str) -> None:
        """Called before a tool is executed.

        Override to add custom logging, validation, or preprocessing.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments passed to the tool.
            tool_call_id: Unique ID for this tool invocation.
        """
        pass

    def on_tool_end(
        self,
        tool_name: str,
        args: dict,
        tool_call_id: str,
        result: Any,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Called after a tool finishes executing.

        Override to add custom logging, caching, or postprocessing.

        Args:
            tool_name: Name of the tool that was called.
            args: Arguments that were passed to the tool.
            tool_call_id: Unique ID for this tool invocation.
            result: The tool's return value (None if failed).
            success: Whether the tool executed successfully.
            error: Error message if failed.
        """
        pass

    def on_model_thinking(self, text: str) -> None:
        """Called when model emits intermediate reasoning text.

        Override to capture or display model's thinking process.

        Args:
            text: The intermediate text from the model.
        """
        pass

    def get_compaction_config(self) -> CompactionConfig:
        """Return the context compaction policy for this agent.

        Override in subclasses to enable or tune compaction.
        """
        return CompactionConfig(enabled=self.ENABLE_COMPACTION)

    def build_compaction_summary_prompt(
        self,
        older_messages: list[Any],
        preserved_messages: list[Any],
        *,
        config: CompactionConfig,
    ) -> tuple[str | None, str]:
        """Build the summarizer prompt for history compaction."""
        transcript_lines = self._transcript_lines_for_compaction(
            older_messages, config.max_message_chars
        )
        transcript_lines = trimmed_transcript_lines(
            transcript_lines, max_chars=config.max_transcript_chars
        )
        transcript = "\n\n".join(transcript_lines)
        if not transcript.strip():
            transcript = self._fallback_compaction_summary(transcript_lines)

        system_prompt = (
            "You compact agent conversation history so the same agent can keep "
            "working after earlier context is removed. Return plain text only."
        )
        prompt = (
            f"Summarize the earlier conversation history for the `{self.name}` agent.\n"
            "Focus on durable context that the next model call still needs:\n"
            "- the user's active goal, constraints, and preferences\n"
            "- important files, URLs, entities, and IDs already found\n"
            "- tool results, decisions, and partial work that should not be repeated\n"
            "- unresolved questions and the most likely next step\n"
            "- anything in the preserved recent messages that depends on older context\n\n"
            "Be concise but information-dense. Do not add preamble, meta commentary, or bullet"
            " numbering unless it helps clarity.\n\n"
            f"Older messages being summarized: {len(older_messages)}\n"
            f"Recent messages preserved verbatim after this summary: {len(preserved_messages)}\n\n"
            "Older transcript:\n"
            f"{transcript}"
        )
        return system_prompt, prompt

    def build_compacted_summary_message(
        self,
        summary: str,
        *,
        preserved_messages: int,
    ) -> Any:
        """Build the synthetic history item that replaces old context."""
        preserved_note = (
            "Recent raw messages continue after this summary and remain authoritative."
            if preserved_messages
            else "No raw messages were preserved after this summary."
        )
        text = (
            f"Internal context compaction summary for the ongoing `{self.name}` agent.\n\n"
            "This summary replaces earlier conversation history so the agent can continue "
            "the same task without losing important context.\n"
            f"{preserved_note}\n\n"
            f"{summary.strip()}"
        )
        return self._provider.build_user_message(text)

    def on_compaction_start(
        self,
        *,
        compaction_id: str,
        scope: str,
        pre_tokens: int,
        config: CompactionConfig,
        history_items_before: int,
    ) -> None:
        """Called before history compaction starts."""
        pass

    def on_compaction_complete(
        self,
        *,
        compaction_id: str,
        scope: str,
        pre_tokens: int,
        post_tokens: int,
        config: CompactionConfig,
        history_items_before: int,
        history_items_after: int,
        older_messages: int,
        preserved_messages: int,
        summary: str,
        duration_ms: int,
        contents: list[Any],
        save_history: bool,
    ) -> None:
        """Called after history compaction succeeds."""
        pass

    def on_compaction_failed(
        self,
        *,
        compaction_id: str,
        scope: str,
        pre_tokens: int,
        config: CompactionConfig,
        history_items_before: int,
        error: Exception,
        duration_ms: int,
        contents: list[Any],
        save_history: bool,
    ) -> None:
        """Called when history compaction fails."""
        pass

    # --- Provider Access ---

    @property
    def client(self):
        """The underlying Gemini client (backward compat).

        Only available when using GeminiProvider.

        Raises:
            AttributeError: If the provider is not GeminiProvider.
        """
        from agent_core.providers.gemini import GeminiProvider
        if isinstance(self._provider, GeminiProvider):
            return self._provider.client
        raise AttributeError(
            f"'client' is only available with GeminiProvider, "
            f"not {type(self._provider).__name__}"
        )

    # --- Event Emission ---

    @property
    def event_bus(self) -> EventBus:
        """The event bus this agent emits to.

        Use this in lifecycle hooks to emit custom events to the same
        bus as the agent's built-in events::

            def on_tool_end(self, tool_name, args, tool_call_id, result, success, error=None):
                if tool_name == "add_papers":
                    self.event_bus.emit(Event(
                        type="papers_added",
                        agent=self.instance_id,
                        details={"count": len(result)},
                    ))
        """
        return self._event_bus

    def _emit_event(self, event_type, **kwargs) -> Event:
        """Emit an event on this agent's event bus."""
        event = Event(type=event_type, **kwargs)
        self._event_bus.emit(event)
        return event

    # --- Cancellation ---

    def cancel(self) -> None:
        """Request cancellation of the current run.

        Thread-safe. The agent will stop after completing any
        in-progress tool executions. The run() or run_stateless()
        call will return "[Cancelled]".
        """
        self._cancel_event.set()

    # --- Tool Registration ---

    def register_tool(self, func: Callable) -> None:
        """Register a tool function for this agent.

        The function should have type hints and a docstring.

        Args:
            func: A callable with type hints and docstring.
        """
        tool_name = getattr(func, "__name__", str(func))
        wrapped = self._wrap_tool_with_events(func)
        self._tools[tool_name] = wrapped
        self._tool_functions.append(func)

        # Rebuild tool schemas via provider
        self._tool_schemas = self._provider.build_tool_schemas(self._tool_functions)

        if self._cache_enabled:
            self._instance_cache_registry.invalidate(self.instance_id)

    def unregister_tool(self, name: str) -> None:
        """Remove a registered tool by name.

        Args:
            name: The function name of the tool to remove.

        Raises:
            KeyError: If no tool with this name is registered.
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered with name: {name}")

        del self._tools[name]

        self._tool_functions = [
            f for f in self._tool_functions
            if getattr(f, "__name__", str(f)) != name
        ]

        self._rebuild_tool_schemas()

        if self._cache_enabled:
            self._instance_cache_registry.invalidate(self.instance_id)

    def _rebuild_tool_schemas(self) -> None:
        """Rebuild tool schemas from current _tool_functions via provider."""
        self._tool_schemas = self._provider.build_tool_schemas(self._tool_functions)

    def _wrap_tool_with_events(self, func: Callable) -> Callable:
        """Wrap a tool function to emit events and call hooks."""
        tool_name = getattr(func, "__name__", str(func))
        # Clean up tool name (remove leading underscore if present)
        display_name = tool_name.lstrip("_")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate unique tool_call_id for pairing START with END
            tool_call_id = f"{display_name}_{uuid.uuid4().hex[:8]}"

            # Build args hint for visualization
            args_hint = None
            full_code = None

            if kwargs:
                for key, val in kwargs.items():
                    if isinstance(val, str) and val:
                        hint_val = val[:30] + "..." if len(val) > 30 else val
                        args_hint = hint_val
                        # For code tools, store full code
                        if display_name in self.CODE_TOOLS and key == "code":
                            full_code = val
                        break

            details = {"args_hint": args_hint} if args_hint else {}
            if full_code:
                details["code"] = full_code

            # Call hook
            self.on_tool_start(display_name, kwargs, tool_call_id)

            # Emit event if enabled
            if self.emit_tool_events:
                self._emit_event(
                    EventType.TOOL_START,
                    agent=self.instance_id,
                    agent_type=self.name,
                    tool=display_name,
                    tool_call_id=tool_call_id,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details=details,
                )

            try:
                result = func(*args, **kwargs)

                # Summarize result for UI
                result_summary = self._summarize_result(result)
                result_type = type(result).__name__

                result_details = {
                    **details,
                    "result_summary": result_summary,
                    "result_type": result_type,
                }

                # Let subclasses enrich event details for domain-specific tools
                self._enrich_tool_event_details(
                    display_name, result, result_details, tool_call_id
                )

                # Call hook
                self.on_tool_end(display_name, kwargs, tool_call_id, result, success=True)

                # Emit event if enabled
                if self.emit_tool_events:
                    self._emit_event(
                        EventType.TOOL_END,
                        agent=self.instance_id,
                        agent_type=self.name,
                        tool=display_name,
                        tool_call_id=tool_call_id,
                        status=EventStatus.COMPLETED,
                        parent_agent=self._parent_agent,
                        wave_id=self._wave_id,
                        details=result_details,
                    )
                return result

            except Exception as e:
                # Call hook
                self.on_tool_end(
                    display_name, kwargs, tool_call_id, None, success=False, error=str(e)
                )

                # Emit event if enabled
                if self.emit_tool_events:
                    self._emit_event(
                        EventType.TOOL_END,
                        agent=self.instance_id,
                        agent_type=self.name,
                        tool=display_name,
                        tool_call_id=tool_call_id,
                        status=EventStatus.FAILED,
                        parent_agent=self._parent_agent,
                        wave_id=self._wave_id,
                        details={"error": str(e), **details},
                    )
                raise

        return wrapper

    # --- Tool Event Detail Enrichment ---

    def _enrich_tool_event_details(
        self,
        tool_name: str,
        result: Any,
        details: dict,
        tool_call_id: str,
    ) -> None:
        """Enrich TOOL_END event details with domain-specific data.

        Override in subclasses to extract domain-specific fields from
        tool results into event details. The base implementation is a no-op.

        Example for a code execution agent::

            def _enrich_tool_event_details(self, tool_name, result, details, tool_call_id):
                if tool_name in self.CODE_TOOLS and isinstance(result, dict):
                    for key in ("code", "stdout", "stderr", "success"):
                        if key in result:
                            details[key] = result[key]

        Args:
            tool_name: Name of the tool that executed.
            result: The tool's return value.
            details: Mutable dict to enrich — will be included in the event.
            tool_call_id: Unique ID for this tool invocation.
        """
        pass

    # --- History Management ---

    def clear_history(self) -> None:
        """Clear conversation history (in-memory and persistent)."""
        self._history = []
        if self._session_id and self._conversation_store:
            self._conversation_store.clear(self._session_id, self.name)
        if self._cache_enabled:
            self._instance_cache_registry.invalidate(self.instance_id)

    def _invalidate_cache(self) -> None:
        """Invalidate the context cache.

        Call after changing system_prompt or tools at runtime.
        """
        if self._cache_enabled:
            self._instance_cache_registry.invalidate(self.instance_id)

    def close(self) -> None:
        """Clean up agent resources (unregisters from cache registry)."""
        if self._cache_enabled:
            self._instance_cache_registry.unregister(self.instance_id)
            self._cache_enabled = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _save_history(self) -> None:
        """Save conversation history to persistent storage."""
        if self._session_id and self._conversation_store:
            self._conversation_store.save(self._session_id, self.name, self._history)

    # --- Result Summarization ---

    def _summarize_result(self, result: Any, max_length: int = 200) -> str:
        """Create a short summary of tool result for UI display.

        Override in subclasses for domain-specific summarization
        (e.g., code execution results with defined_variables).
        """
        if result is None:
            return "None"

        if isinstance(result, dict):
            if result.get("error"):
                error_msg = str(result["error"])[:100]
                return f"Error: {error_msg}"

            if "text" in result:
                text = str(result["text"])
                return text[:max_length] + "..." if len(text) > max_length else text

            if "stdout" in result and result["stdout"]:
                stdout = str(result["stdout"])
                return stdout[:max_length] + "..." if len(stdout) > max_length else stdout

            if "output" in result:
                output = str(result["output"])
                return output[:max_length] + "..." if len(output) > max_length else output

            if "result" in result:
                res = str(result["result"])
                return res[:max_length] + "..." if len(res) > max_length else res

            keys = list(result.keys())[:5]
            extra = f" (+{len(result) - 5} more)" if len(result) > 5 else ""
            return f"Dict: {{{', '.join(keys)}{extra}}}"

        if isinstance(result, str):
            return result[:max_length] + "..." if len(result) > max_length else result

        if isinstance(result, (list, tuple)):
            return f"List[{len(result)} items]"

        if isinstance(result, bool):
            return str(result)

        if isinstance(result, (int, float)):
            return str(result)

        result_str = str(result)
        return result_str[:max_length] + "..." if len(result_str) > max_length else result_str

    # --- Context Information ---

    def get_context(self) -> dict:
        """Get the full context window that the agent sees.

        Returns a dict with system prompt and conversation history,
        useful for debugging and visibility into agent state.
        """
        history_formatted = [
            entry
            for msg in self._history
            if (entry := self._provider.format_message_for_display(msg)) is not None
        ]

        tool_names = list(self._tools.keys())
        context_tokens = self._count_context_tokens()

        return {
            "system_prompt": self.system_prompt,
            "history": history_formatted,
            "tools": tool_names,
            "agent_type": self.name,
            "instance_id": self.instance_id,
            "context_tokens": context_tokens,
        }

    def _count_context_tokens(self) -> int:
        """Count tokens in the current context window."""
        try:
            return self._provider.count_tokens(
                self.model_name, self._history, self.system_prompt
            )
        except Exception:
            return 0

    def _count_context_tokens_for_compaction(
        self, contents: list[Any], max_chars: int
    ) -> int:
        """Count tokens for an arbitrary message list, with a display fallback."""
        try:
            token_count = self._provider.count_tokens(
                self.model_name, contents, self.system_prompt
            )
        except Exception:
            token_count = 0
        if token_count:
            return token_count
        return estimate_history_tokens(
            self._provider, contents, max_chars=max_chars
        ) + approximate_tokens(self.system_prompt or "")

    def _transcript_lines_for_compaction(
        self, messages: list[Any], max_chars: int
    ) -> list[str]:
        lines = []
        for message in messages:
            line = render_message_for_compaction(
                self._provider, message, max_chars=max_chars
            )
            if line:
                lines.append(line)
        return lines

    def _fallback_compaction_summary(self, transcript_lines: list[str]) -> str:
        if not transcript_lines:
            return "No earlier transcript content was available to summarize."
        tail = transcript_lines[-6:]
        return "Fallback compacted context:\n" + "\n\n".join(tail)

    def _generate_compaction_summary(
        self,
        older_messages: list[Any],
        preserved_messages: list[Any],
        *,
        config: CompactionConfig,
    ) -> tuple[str, TokenUsage]:
        system_prompt, prompt = self.build_compaction_summary_prompt(
            older_messages,
            preserved_messages,
            config=config,
        )
        raw_response = self._generate_with_retry(
            model=self.model_name,
            messages=[self._provider.build_user_message(prompt)],
            system_prompt=system_prompt,
            temperature=0.2,
            max_output_tokens=config.summary_max_output_tokens,
            tool_schemas=None,
            cache_config=None,
        )
        if raw_response is None:
            raise RuntimeError("Compaction cancelled during summary generation")

        parsed = self._provider.parse_response(raw_response)
        summary = (parsed.text or "").strip()
        if not summary:
            transcript_lines = self._transcript_lines_for_compaction(
                older_messages, config.max_message_chars
            )
            summary = self._fallback_compaction_summary(transcript_lines)
        return summary, parsed.usage

    def _compaction_scope(self, contents: list[Any], save_history: bool) -> str:
        return "session" if save_history and contents is self._history else "run"

    def _maybe_compact_history(
        self,
        contents: list[Any],
        *,
        save_history: bool,
    ) -> tuple[bool, TokenUsage]:
        config = self.get_compaction_config()
        if not config.enabled or self._compaction_count >= config.max_compactions_per_run:
            return False, TokenUsage()
        invalid_fields = [
            field
            for field in (
                "trigger_tokens",
                "tail_token_budget",
                "summary_max_output_tokens",
                "max_transcript_chars",
                "max_message_chars",
                "max_compactions_per_run",
            )
            if getattr(config, field) <= 0
        ]
        if invalid_fields:
            logger.warning(
                "Compaction disabled for agent %s: invalid config fields: %s",
                self.name,
                ", ".join(invalid_fields),
            )
            return False, TokenUsage()
        if len(contents) < max(2, config.min_preserved_messages + 1):
            return False, TokenUsage()

        pre_tokens = self._count_context_tokens_for_compaction(
            contents, config.max_message_chars
        )
        if not pre_tokens or pre_tokens < config.trigger_tokens:
            return False, TokenUsage()

        preserved_start = select_preserved_tail_start(
            self._provider,
            contents,
            tail_token_budget=config.tail_token_budget,
            min_messages=config.min_preserved_messages,
            max_chars=config.max_message_chars,
        )
        if preserved_start <= 0:
            return False, TokenUsage()

        older_messages = contents[:preserved_start]
        preserved_messages = contents[preserved_start:]
        if not older_messages or not preserved_messages:
            return False, TokenUsage()

        scope = self._compaction_scope(contents, save_history)
        compaction_id = f"compaction_{uuid.uuid4().hex[:8]}"
        started_at = time.time()
        history_items_before = len(contents)
        self._compaction_count += 1
        self.on_compaction_start(
            compaction_id=compaction_id,
            scope=scope,
            pre_tokens=pre_tokens,
            config=config,
            history_items_before=history_items_before,
        )

        try:
            summary, usage = self._generate_compaction_summary(
                older_messages,
                preserved_messages,
                config=config,
            )
            compacted_message = self.build_compacted_summary_message(
                summary,
                preserved_messages=len(preserved_messages),
            )
            contents[:] = [compacted_message, *preserved_messages]
            if save_history:
                self._save_history()
            self._invalidate_cache()

            post_tokens = self._count_context_tokens_for_compaction(
                contents, config.max_message_chars
            )
            self.on_compaction_complete(
                compaction_id=compaction_id,
                scope=scope,
                pre_tokens=pre_tokens,
                post_tokens=post_tokens,
                config=config,
                history_items_before=history_items_before,
                history_items_after=len(contents),
                older_messages=len(older_messages),
                preserved_messages=len(preserved_messages),
                summary=summary,
                duration_ms=int((time.time() - started_at) * 1000),
                contents=contents,
                save_history=save_history,
            )
            return True, usage
        except Exception as exc:
            logger.warning(
                "Compaction failed for agent %s: %s",
                self.name,
                exc,
            )
            self.on_compaction_failed(
                compaction_id=compaction_id,
                scope=scope,
                pre_tokens=pre_tokens,
                config=config,
                history_items_before=history_items_before,
                error=exc,
                duration_ms=int((time.time() - started_at) * 1000),
                contents=contents,
                save_history=save_history,
            )
            return False, TokenUsage()

    # --- Tool Execution ---

    def _execute_tool(self, tool_call: ToolCall) -> tuple[str, Any]:
        """Execute a single tool and return (name, result)."""
        func_name = tool_call.name
        args = tool_call.args

        tool_func = self._tools.get(func_name)
        if not tool_func:
            return func_name, {"error": f"Unknown function: {func_name}"}

        try:
            result = tool_func(**args)
            return func_name, result
        except Exception as e:
            logger.exception("Tool %s raised an exception", func_name)
            return func_name, {"error": str(e)}

    def _execute_tools_parallel(
        self, tool_calls: list[ToolCall]
    ) -> list[tuple[str, Any]]:
        """Execute multiple tool calls in parallel.

        Checks ``_cancel_event`` between completions so a hung tool
        doesn't block cancellation indefinitely.  Completed results are
        always collected; cancelled/pending tools get an error placeholder.
        """
        results: list[tuple[str, Any] | None] = [None] * len(tool_calls)
        executor = ThreadPoolExecutor(
            max_workers=min(len(tool_calls), self.MAX_PARALLEL_TOOLS)
        )
        cancelled = False

        try:
            future_to_idx = {
                executor.submit(self._execute_tool, tc): idx
                for idx, tc in enumerate(tool_calls)
            }

            remaining = set(future_to_idx.keys())
            while remaining:
                # Poll with short timeout so we can check cancellation
                done, remaining = wait(remaining, timeout=0.5, return_when=FIRST_COMPLETED)

                for future in done:
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        func_name = tool_calls[idx].name
                        results[idx] = (func_name, {"error": str(e)})

                if self._cancel_event.is_set() and remaining:
                    # Cancel pending futures and fill placeholders
                    cancelled = True
                    for future in remaining:
                        future.cancel()
                        idx = future_to_idx[future]
                        func_name = tool_calls[idx].name
                        results[idx] = (func_name, {"error": "Cancelled"})
                    break
        finally:
            executor.shutdown(wait=not cancelled, cancel_futures=cancelled)

        for idx, result in enumerate(results):
            if result is None:
                results[idx] = (tool_calls[idx].name, {"error": "Cancelled"})

        return results

    # --- Retry Logic ---

    # Retry config for quota-exhausted (429) errors
    RETRY_MAX_ATTEMPTS: ClassVar[int] = 5
    RETRY_BASE_DELAY: ClassVar[float] = 2.0   # seconds
    RETRY_MAX_DELAY: ClassVar[float] = 60.0    # seconds

    def _generate_with_retry(
        self,
        *,
        stream: bool = False,
        on_text_delta: Callable[[str], None] | None = None,
        **kwargs,
    ):
        """Call provider.generate with exponential backoff on retryable errors.

        Checks _cancel_event during backoff so cancellation is responsive
        even during retry waits.
        """
        use_stream = (
            stream
            and hasattr(self._provider, "generate_stream")
        )
        generate = self._provider.generate_stream if use_stream else self._provider.generate
        if use_stream:
            kwargs["on_text_delta"] = on_text_delta

        for attempt in range(1, self.RETRY_MAX_ATTEMPTS + 1):
            try:
                return generate(**kwargs)
            except Exception as e:
                if self._provider.is_retryable_error(e) and attempt < self.RETRY_MAX_ATTEMPTS:
                    delay = self._provider.get_retry_delay(
                        e, attempt, self.RETRY_BASE_DELAY, self.RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "[%s] Retryable error (attempt %d/%d), retrying in %.1fs: %s",
                        self.name, attempt, self.RETRY_MAX_ATTEMPTS, delay, e,
                    )
                    # Use cancel event as sleep — wakes immediately on cancel
                    if self._cancel_event.wait(timeout=delay):
                        return None  # Caller checks cancel_event
                else:
                    raise

    # --- Main Execution Loop ---

    def _run_with_function_loop(
        self,
        contents: list[Any],
        temperature: float,
        max_output_tokens: int,
        cache_config: dict | None = None,
        save_history: bool = True,
        streaming: bool = False,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, dict]:
        """Run the model with manual function calling loop.

        Args:
            contents: Full conversation history (appended to during loop).
            temperature: Sampling temperature.
            max_output_tokens: Maximum tokens in response.
            cache_config: Optional caching hints for the provider.
                Gemini uses ``{"cache_name": str, "contents_offset": int}``.
            save_history: Whether to persist history after each append.
                True for stateful run(), False for run_stateless().
            streaming: Whether to use provider streaming transport.
            on_text_delta: Optional callback for provider text deltas.
        """
        iteration = 0
        total = TokenUsage()
        last_prompt = 0
        safe_on_text_delta = None
        if on_text_delta is not None:
            def _safe_on_text_delta(delta: str) -> None:
                if not delta:
                    return
                try:
                    on_text_delta(delta)
                except Exception as exc:
                    logger.warning(
                        "[%s] text delta callback failed: %s",
                        self.name,
                        exc,
                    )
            safe_on_text_delta = _safe_on_text_delta

        def _usage_dict() -> dict:
            offset = (cache_config or {}).get("contents_offset", 0)
            return {
                "prompt_tokens": total.prompt_tokens,
                "completion_tokens": total.completion_tokens,
                "total_tokens": total.prompt_tokens + total.completion_tokens,
                "cached_tokens": total.cached_tokens,
                "cache_write_tokens": total.cache_write_tokens,
                "cache_type": (
                    ("explicit" if offset > 0 else "implicit")
                    if total.cached_tokens else None
                ),
                "last_prompt_token_count": last_prompt,
                "model": self.model_name,
            }

        while iteration < self.MAX_ITERATIONS:
            iteration += 1

            # Check for cancellation before each API call
            if self._cancel_event.is_set():
                return "[Cancelled]", _usage_dict()

            while True:
                compacted, compaction_usage = self._maybe_compact_history(
                    contents,
                    save_history=save_history,
                )
                total.prompt_tokens += compaction_usage.prompt_tokens
                total.completion_tokens += compaction_usage.completion_tokens
                total.cached_tokens += compaction_usage.cached_tokens
                total.cache_write_tokens += compaction_usage.cache_write_tokens
                if not compacted:
                    break
                cache_config = None
                if self._cancel_event.is_set():
                    return "[Cancelled]", _usage_dict()

            try:
                raw_response = self._generate_with_retry(
                    model=self.model_name,
                    messages=contents,
                    system_prompt=self.system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    tool_schemas=self._tool_schemas,
                    cache_config=cache_config,
                    stream=streaming,
                    on_text_delta=safe_on_text_delta,
                )
            except Exception as exc:
                overflow_markers = (
                    "context",
                    "prompt",
                    "token",
                    "too large",
                    "too long",
                )
                message = str(exc).lower()
                if any(marker in message for marker in overflow_markers):
                    compacted, compaction_usage = self._maybe_compact_history(
                        contents,
                        save_history=save_history,
                    )
                    total.prompt_tokens += compaction_usage.prompt_tokens
                    total.completion_tokens += compaction_usage.completion_tokens
                    total.cached_tokens += compaction_usage.cached_tokens
                    total.cache_write_tokens += compaction_usage.cache_write_tokens
                    if compacted:
                        cache_config = None
                        iteration -= 1
                        continue
                raise

            # None means cancelled during retry backoff
            if raw_response is None:
                return "[Cancelled]", _usage_dict()

            parsed = self._provider.parse_response(raw_response)

            # Accumulate usage
            last_prompt = parsed.usage.prompt_tokens
            total.prompt_tokens += parsed.usage.prompt_tokens
            total.completion_tokens += parsed.usage.completion_tokens
            total.cached_tokens += parsed.usage.cached_tokens
            total.cache_write_tokens += parsed.usage.cache_write_tokens

            if last_prompt:
                offset = (cache_config or {}).get("contents_offset", 0)
                cache_label = "explicit" if offset > 0 else "implicit"
                pct = (
                    parsed.usage.cached_tokens * 100 // max(last_prompt, 1)
                    if parsed.usage.cached_tokens else 0
                )
                logger.info(
                    "[%s] iter=%d cache=%s: %d cached / %d prompt tokens (%d%%)",
                    self.name, iteration, cache_label,
                    parsed.usage.cached_tokens, last_prompt, pct,
                )
            if parsed.usage.cache_write_tokens:
                logger.info(
                    "[%s] iter=%d cache write: %d prompt tokens",
                    self.name,
                    iteration,
                    parsed.usage.cache_write_tokens,
                )

            # Emit intermediate thinking text alongside tool calls
            if parsed.thinking_text and parsed.tool_calls:
                self.on_model_thinking(parsed.thinking_text)
                if self.emit_lifecycle_events:
                    self._emit_event(
                        EventType.MODEL_THINKING,
                        agent=self.instance_id,
                        agent_type=self.name,
                        parent_agent=self._parent_agent,
                        wave_id=self._wave_id,
                        details={
                            "text": parsed.thinking_text,
                            "streamed": parsed.streamed_text,
                        },
                    )

            if not parsed.tool_calls:
                # Final text response
                if parsed.raw_message is not None:
                    contents.append(parsed.raw_message)
                if save_history:
                    self._save_history()
                return parsed.text or "", _usage_dict()

            # Append the model's tool-calling message
            if parsed.raw_message is not None:
                contents.append(parsed.raw_message)
            if save_history:
                self._save_history()

            # Execute tools
            results = self._execute_tools_parallel(parsed.tool_calls)

            # Build and append tool result messages
            tool_msgs = self._provider.build_tool_result_messages(
                parsed.tool_calls, results,
            )
            if isinstance(tool_msgs, list):
                contents.extend(tool_msgs)
            else:
                contents.append(tool_msgs)
            if save_history:
                self._save_history()

            # Check for cancellation after tool execution
            if self._cancel_event.is_set():
                return "[Cancelled]", _usage_dict()

            # Mid-loop: notify registry (may fire new cache) and re-query
            if self._cache_enabled:
                self._instance_cache_registry.notify(
                    self.instance_id,
                    contents,
                    self.system_prompt,
                    self._tool_functions or None,
                    token_count=last_prompt,
                )
                advice = self._instance_cache_registry.get_advice(
                    self.instance_id, self.system_prompt, self._tool_functions
                )
                if advice.cache_name:
                    cache_config = {
                        "cache_name": advice.cache_name,
                        "contents_offset": advice.contents_offset,
                    }
                    logger.debug(
                        "Mid-loop cache switch: offset=%d",
                        advice.contents_offset,
                    )

            # Emit context update
            if self.emit_lifecycle_events:
                self._emit_event(
                    EventType.CONTEXT_UPDATE,
                    agent=self.instance_id,
                    agent_type=self.name,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details={"context_tokens": last_prompt},
                )

        return f"[Max iterations ({self.MAX_ITERATIONS}) reached]", _usage_dict()

    def _build_cache_config(
        self,
        wait_for_cache: bool = False,
    ) -> dict | None:
        """Query cache registry and return cache_config dict or None.

        Args:
            wait_for_cache: If True, block until any pending cache creation
                completes before returning.
        """
        if not self._cache_enabled:
            return None
        advice = self._instance_cache_registry.get_advice(
            self.instance_id, self.system_prompt, self._tool_functions,
            wait=wait_for_cache,
        )
        if advice.cache_name:
            return {
                "cache_name": advice.cache_name,
                "contents_offset": advice.contents_offset,
            }
        return None

    # --- Run Execution ---

    _CANCELLED_RESULT = "[Cancelled]"

    def _execute_run(self, prompt: str, execute_fn: Callable[[], tuple[str, dict]]) -> str:
        """Shared execution wrapper for run() and run_stateless().

        Handles: cancel-clear, hooks, event emission, error handling.

        Args:
            prompt: The prompt string (for hooks and events).
            execute_fn: Callable that runs the actual generation loop
                       and returns (result_text, token_usage).
        """
        if self._owns_cancel_event:
            self._cancel_event.clear()

        self._compaction_count = 0
        self.on_agent_start(prompt)

        if self.emit_lifecycle_events:
            self._emit_event(
                EventType.AGENT_START,
                agent=self.instance_id,
                agent_type=self.name,
                parent_agent=self._parent_agent,
                wave_id=self._wave_id,
                details={"prompt": prompt, "model": self.model_name},
            )

        try:
            result, token_usage = execute_fn()
            was_cancelled = result == self._CANCELLED_RESULT

            self.on_agent_end(result, success=not was_cancelled, cancelled=was_cancelled)

            if self.emit_lifecycle_events:
                self._emit_event(
                    EventType.AGENT_END,
                    agent=self.instance_id,
                    agent_type=self.name,
                    status=EventStatus.COMPLETED,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details={"result": result, "token_usage": token_usage},
                )

            return result

        except Exception as e:
            self.on_agent_end("", success=False, error=str(e))

            if self.emit_lifecycle_events:
                self._emit_event(
                    EventType.AGENT_END,
                    agent=self.instance_id,
                    agent_type=self.name,
                    status=EventStatus.FAILED,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details={"error": str(e)},
                )
            raise

    def run(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 32768,
        wait_for_cache: bool = False,
        streaming: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Run the agent with a prompt (maintains conversation history).

        Args:
            prompt: User prompt or task description.
            temperature: Sampling temperature.
            max_output_tokens: Maximum tokens in response.
            wait_for_cache: If True, block until any pending cache creation
                completes before the first generate call. Trades latency for
                guaranteed cache savings on large contexts.
            streaming: If True, use provider streaming when available. If
                None, uses this agent's configured streaming default.
            on_text_delta: Callback invoked with incremental text chunks.

        Returns:
            The agent's final text response.
        """

        def _execute() -> tuple[str, dict]:
            use_streaming = self.streaming if streaming is None else streaming
            self._history.append(self._provider.build_user_message(prompt))
            self._save_history()

            cache_config = self._build_cache_config(wait_for_cache=wait_for_cache)

            try:
                result, token_usage = self._run_with_function_loop(
                    self._history, temperature, max_output_tokens,
                    cache_config=cache_config,
                    streaming=use_streaming,
                    on_text_delta=on_text_delta if use_streaming else None,
                )
            except Exception as e:
                if cache_config and self._cache_enabled:
                    logger.warning("Cached call failed, falling back: %s", e)
                    self._instance_cache_registry.invalidate(self.instance_id)
                    result, token_usage = self._run_with_function_loop(
                        self._history, temperature, max_output_tokens,
                        streaming=use_streaming,
                        on_text_delta=on_text_delta if use_streaming else None,
                    )
                else:
                    raise

            if self._cache_enabled:
                self._instance_cache_registry.notify(
                    self.instance_id,
                    self._history,
                    self.system_prompt,
                    self._tool_functions or None,
                    token_count=token_usage.get("last_prompt_token_count"),
                )

            return result, token_usage

        return self._execute_run(prompt, _execute)

    def run_stateless(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 32768,
        streaming: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Run the agent without maintaining conversation history.

        Useful for one-shot tasks where each call is independent.

        Args:
            prompt: User prompt or task description.
            context: Optional context dict to include in the prompt.
            temperature: Sampling temperature.
            max_output_tokens: Maximum tokens in response.
            streaming: If True, use provider streaming when available. If
                None, uses this agent's configured streaming default.
            on_text_delta: Callback invoked with incremental text chunks.

        Returns:
            The agent's final text response.
        """
        full_prompt = prompt
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            full_prompt = f"Context:\n{context_str}\n\nTask: {prompt}"

        def _execute() -> tuple[str, dict]:
            contents = [self._provider.build_user_message(full_prompt)]
            use_streaming = self.streaming if streaming is None else streaming
            return self._run_with_function_loop(
                contents,
                temperature,
                max_output_tokens,
                save_history=False,
                streaming=use_streaming,
                on_text_delta=on_text_delta if use_streaming else None,
            )

        return self._execute_run(full_prompt, _execute)


def agent_as_tool(agent: Agent, description: str | None = None) -> Callable:
    """Wrap an agent as a tool function for another agent.

    This allows a parent agent to delegate tasks to sub-agents.

    Args:
        agent: The agent to wrap.
        description: Optional description override.

    Returns:
        A callable that can be registered as a tool.
    """

    def tool_func(task: str, context: str = "") -> str:
        """Delegate a task to a specialized agent.

        Args:
            task: The task description for the agent.
            context: Additional context as a string (e.g., JSON).

        Returns:
            The agent's response.
        """
        ctx = None
        if context:
            try:
                import json

                ctx = json.loads(context)
            except json.JSONDecodeError:
                ctx = {"raw_context": context}

        return agent.run_stateless(task, context=ctx)

    tool_func.__name__ = f"{agent.name}_agent"
    tool_func.__doc__ = (
        description
        or f"Delegate task to the {agent.name} specialist agent. {agent.system_prompt[:200]}"
    )

    return tool_func
