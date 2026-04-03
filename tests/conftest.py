"""Shared fixtures and mocks for agent_core tests."""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# --- Mock classes for google.genai SDK ---


class MockFunctionCall:
    """Mock for google.genai.types.FunctionCall."""

    def __init__(self, name: str, args: dict | None = None):
        self.name = name
        self.args = args or {}


class MockFunctionResponse:
    """Mock for google.genai.types.FunctionResponse."""

    def __init__(self, name: str, response: dict):
        self.name = name
        self.response = response


class MockPart:
    """Mock for google.genai.types.Part."""

    def __init__(
        self,
        text: str = None,
        function_call: MockFunctionCall = None,
        function_response: MockFunctionResponse = None,
        thought: str = None,
        inline_data: Any = None,
    ):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response
        self.thought = thought
        self.inline_data = inline_data

    @classmethod
    def from_text(cls, text: str):
        return cls(text=text)

    @classmethod
    def from_function_call(cls, name: str, args: dict):
        return cls(function_call=MockFunctionCall(name, args))

    @classmethod
    def from_function_response(cls, name: str, response: dict):
        return cls(function_response=MockFunctionResponse(name, response))


class MockContent:
    """Mock for google.genai.types.Content."""

    def __init__(self, role: str, parts: list):
        self.role = role
        self.parts = parts


class MockCandidate:
    """Mock for response candidate."""

    def __init__(self, content: MockContent):
        self.content = content


class MockUsageMetadata:
    """Mock for response usage metadata."""

    def __init__(self, prompt_token_count: int = 100, candidates_token_count: int = 50, cached_content_token_count: int = 0):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.cached_content_token_count = cached_content_token_count


class MockTokenCountResponse:
    """Mock for count_tokens response."""

    def __init__(self, total_tokens: int = 1000):
        self.total_tokens = total_tokens


class MockCachedContent:
    """Mock for CachedContent returned by caches.create."""

    def __init__(self, name: str = "cachedContents/test-cache-123"):
        self.name = name


class MockResponse:
    """Mock for generate_content response."""

    def __init__(
        self,
        text: str,
        content: MockContent = None,
        usage_metadata: MockUsageMetadata = None,
    ):
        self._text = text
        self._content = content
        self.candidates = [MockCandidate(content)] if content else []
        self.usage_metadata = usage_metadata or MockUsageMetadata()

    @property
    def text(self):
        return self._text


def make_text_response(text: str) -> MockResponse:
    """Create a simple text-only response (no tool calls)."""
    content = MockContent(role="model", parts=[MockPart(text=text)])
    return MockResponse(text, content)


def make_tool_call_response(
    tool_name: str, args: dict, thinking_text: str = None
) -> MockResponse:
    """Create a response with a function call."""
    parts = []
    if thinking_text:
        parts.append(MockPart(text=thinking_text))
    parts.append(MockPart(function_call=MockFunctionCall(tool_name, args)))
    content = MockContent(role="model", parts=parts)
    return MockResponse("", content)


def make_multi_tool_call_response(calls: list[tuple[str, dict]]) -> MockResponse:
    """Create a response with multiple parallel function calls."""
    parts = [MockPart(function_call=MockFunctionCall(name, args)) for name, args in calls]
    content = MockContent(role="model", parts=parts)
    return MockResponse("", content)


# --- Fixtures ---


@pytest.fixture
def mock_env():
    """Set required environment variables."""
    with patch.dict(os.environ, {"GOOGLE_PROJECT_ID": "test-project"}):
        yield


@pytest.fixture
def mock_genai():
    """Mock the google.genai module for agent_core (base + provider)."""
    with (
        patch("agent_core.agents.base.genai") as mock,
        patch("agent_core.providers.gemini.genai", mock),
        patch("agent_core.providers.gemini.types") as mock_types,
        patch("agent_core.providers.gemini.ClientError", type("MockClientError", (Exception,), {"code": None})),
    ):
        # Setup default client behavior
        mock_client = MagicMock()
        mock.Client.return_value = mock_client

        # Default count_tokens returns below cache threshold
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(1000)

        # Default caches behavior
        mock_client.caches.create.return_value = MockCachedContent()
        mock_client.caches.delete.return_value = None

        # Wire up types to use our mock classes
        mock_types.Content = MockContent
        mock_types.Part = MockPart
        mock_types.Tool = MagicMock
        mock_types.FunctionDeclaration = MagicMock()
        mock_types.FunctionDeclaration.from_callable.return_value = MagicMock()
        mock_types.GenerateContentConfig = MagicMock
        mock_types.AutomaticFunctionCallingConfig = MagicMock

        yield mock


@pytest.fixture
def mock_types():
    """Mock the google.genai.types module for agent_core (legacy fixture)."""
    with patch("agent_core.providers.gemini.types") as mock:
        mock.Content = MockContent
        mock.Part = MockPart
        mock.GenerateContentConfig = MagicMock
        mock.AutomaticFunctionCallingConfig = MagicMock
        yield mock


@pytest.fixture
def mock_events():
    """Mock event emission."""
    with patch("agent_core.agents.base.emit_event") as mock:
        yield mock


@pytest.fixture
def mock_cache_registry():
    """Mock the ContextCacheRegistry to isolate Agent tests from caching logic."""
    from agent_core.core.caching import CacheAdvice

    with patch("agent_core.agents.base.ContextCacheRegistry") as mock_cls:
        mock_reg = MagicMock()
        mock_reg.get_advice.return_value = CacheAdvice(cache_name=None, contents_offset=0)
        mock_cls.return_value = mock_reg
        yield mock_reg


@pytest.fixture
def mock_client():
    """Create a pre-configured mock Gemini client for injection tests."""
    client = MagicMock()
    client.models.count_tokens.return_value = MockTokenCountResponse(1000)
    client.models.generate_content.return_value = make_text_response("ok")
    client.caches.create.return_value = MockCachedContent()
    client.caches.delete.return_value = None
    return client


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)
