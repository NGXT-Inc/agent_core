"""LLM provider abstractions for agent_core.

This module defines the provider-neutral types and protocol that allow
agents to work with different LLM backends (Gemini, OpenAI, etc.).
"""

from agent_core.providers.types import (
    AgentResponse,
    FilePart,
    FileOutputPart,
    LLMProvider,
    MessagePart,
    OutputPart,
    ParsedResponse,
    ProviderCapabilities,
    TextPart,
    TextOutputPart,
    TokenUsage,
    ToolCall,
    UnsupportedInputPart,
    UserMessage,
    UserMessageInput,
    coerce_user_message,
)

__all__ = [
    "LLMProvider",
    "AgentResponse",
    "FilePart",
    "FileOutputPart",
    "MessagePart",
    "OutputPart",
    "ParsedResponse",
    "ProviderCapabilities",
    "TextPart",
    "TextOutputPart",
    "TokenUsage",
    "ToolCall",
    "UnsupportedInputPart",
    "UserMessage",
    "UserMessageInput",
    "coerce_user_message",
]

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
