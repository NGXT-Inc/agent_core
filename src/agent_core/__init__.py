"""Agent Core - Extensible agent orchestration framework with multi-LLM support."""

from agent_core.agents.base import Agent, agent_as_tool, generate_instance_id
from agent_core.core.events import (
    EventType,
    EventStatus,
    Event,
    EventBus,
    get_event_bus,
    emit_event,
)
from agent_core.providers import (
    LLMProvider,
    ToolCall,
    TokenUsage,
    ParsedResponse,
)
from agent_core.providers.gemini import GeminiProvider

__all__ = [
    # Agent
    "Agent",
    "agent_as_tool",
    "generate_instance_id",
    # Providers
    "LLMProvider",
    "GeminiProvider",
    "ToolCall",
    "TokenUsage",
    "ParsedResponse",
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
