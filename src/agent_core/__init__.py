"""Agent Core - Extensible agent orchestration framework with multi-LLM support."""

from agent_core.agents.base import (
    Agent,
    EmptyModelResponseError,
    agent_as_tool,
    generate_instance_id,
)
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
    "EmptyModelResponseError",
    "agent_as_tool",
    "generate_instance_id",
    # Providers
    "LLMProvider",
    "AgentResponse",
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
