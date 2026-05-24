"""Agent Core - Extensible agent orchestration framework with multi-LLM support."""

import atexit as _atexit

from agent_core import _native as _native_ext


# Tear down the C++ subsystems (Registry reaper, SQLite writer) while the
# Python interpreter is still in a healthy state. Without this, static-local
# destructors fire after Python's shutdown which can segfault when SessionHandle
# destructors call back into the registry.
@_atexit.register
def _shutdown_native_extension() -> None:  # pragma: no cover — atexit-only
    try:
        _native_ext._shutdown()
    except Exception:
        pass


from agent_core import registry
from agent_core.agents.base import Agent, agent_as_tool, generate_instance_id
from agent_core.agents.compaction import CompactionConfig
from agent_core.core.events import (
    EventType,
    EventStatus,
    Event,
    EventBus,
    get_event_bus,
    emit_event,
)
from agent_core.providers import (
    AgentResponse,
    CanonicalMessage,
    CanonicalRole,
    FilePart,
    FileOutputPart,
    LLMProvider,
    MessagePart,
    OutputPart,
    ProviderCapabilities,
    ToolCall,
    TokenUsage,
    ParsedResponse,
    TextPart,
    TextOutputPart,
    UnsupportedInputPart,
    UserMessage,
    UserMessageInput,
    coerce_user_message,
)
from agent_core.providers.gemini import GeminiProvider

__all__ = [
    # Agent
    "Agent",
    "CompactionConfig",
    "agent_as_tool",
    "generate_instance_id",
    # Providers
    "LLMProvider",
    "AgentResponse",
    "CanonicalMessage",
    "CanonicalRole",
    "FilePart",
    "FileOutputPart",
    "GeminiProvider",
    "MessagePart",
    "OutputPart",
    "ProviderCapabilities",
    "TextPart",
    "TextOutputPart",
    "ToolCall",
    "TokenUsage",
    "ParsedResponse",
    "UnsupportedInputPart",
    "UserMessage",
    "UserMessageInput",
    "coerce_user_message",
    # Events
    "EventType",
    "EventStatus",
    "Event",
    "EventBus",
    "get_event_bus",
    "emit_event",
    # Registry
    "registry",
]

# Conditional import — openai is an optional dependency
try:
    from agent_core.providers.openai import OpenAIProvider
    __all__.append("OpenAIProvider")
except ImportError:
    pass

try:
    from agent_core.providers.openrouter import OpenRouterCacheConfig, OpenRouterProvider
    __all__.extend(["OpenRouterProvider", "OpenRouterCacheConfig"])
except ImportError:
    pass
