"""Tests for provider implementations."""

import json
from typing import Optional
from unittest.mock import MagicMock

import pytest

from agent_core.providers.types import LLMProvider, ParsedResponse, TokenUsage, ToolCall


# ============================================================
# Schema generation tests (_openai_schema)
# ============================================================


class TestOpenAISchemaGeneration:
    """Test inspect-based JSON Schema from Python callables."""

    def test_simple_function(self):
        from agent_core.providers._openai_schema import callable_to_openai_tool

        def greet(name: str) -> str:
            """Say hello to someone.

            Args:
                name: The person's name.
            """
            return f"Hello {name}"

        schema = callable_to_openai_tool(greet)
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "greet"
        assert "hello" in fn["description"].lower()
        params = fn["parameters"]
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["name"]["description"] == "The person's name."
        assert "name" in params["required"]

    def test_multiple_types(self):
        from agent_core.providers._openai_schema import callable_to_openai_tool

        def compute(x: int, y: float, flag: bool) -> str:
            """Compute something."""
            return ""

        schema = callable_to_openai_tool(compute)
        props = schema["function"]["parameters"]["properties"]
        assert props["x"]["type"] == "integer"
        assert props["y"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"

    def test_optional_param(self):
        from agent_core.providers._openai_schema import callable_to_openai_tool

        def search(query: str, limit: int = 10) -> list:
            """Search for items."""
            return []

        schema = callable_to_openai_tool(search)
        params = schema["function"]["parameters"]
        assert "query" in params["required"]
        assert "limit" not in params.get("required", [])

    def test_list_type(self):
        from agent_core.providers._openai_schema import callable_to_openai_tool

        def process(items: list[str]) -> str:
            """Process items."""
            return ""

        schema = callable_to_openai_tool(process)
        prop = schema["function"]["parameters"]["properties"]["items"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_optional_type_hint(self):
        from agent_core.providers._openai_schema import callable_to_openai_tool

        def lookup(key: str, default: Optional[str] = None) -> str:
            """Look up a key."""
            return ""

        schema = callable_to_openai_tool(lookup)
        props = schema["function"]["parameters"]["properties"]
        assert props["default"]["type"] == "string"

    def test_no_docstring(self):
        from agent_core.providers._openai_schema import callable_to_openai_tool

        def bare(x: str) -> str:
            return x

        schema = callable_to_openai_tool(bare)
        assert schema["function"]["name"] == "bare"
        assert schema["function"]["description"] == ""


# ============================================================
# OpenAIProvider tests
# ============================================================


class TestOpenAIProvider:
    """Test OpenAIProvider with mocked OpenAI client."""

    def _make_provider(self, **kwargs):
        from agent_core.providers.openai import OpenAIProvider
        mock_client = MagicMock()
        return OpenAIProvider(client=mock_client, **kwargs), mock_client

    def _make_mock_response(
        self,
        content="Hello",
        tool_calls=None,
        reasoning_details=None,
        prompt_tokens=100,
        completion_tokens=50,
    ):
        """Build a mock OpenAI response."""
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = content
        msg.tool_calls = tool_calls
        msg.reasoning_details = reasoning_details

        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens

        choice = MagicMock()
        choice.message = msg

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        return response

    def _make_mock_tool_call(self, tc_id, name, arguments):
        tc = MagicMock()
        tc.id = tc_id
        tc.function.name = name
        tc.function.arguments = json.dumps(arguments)
        return tc

    def test_generate_prepends_system_prompt(self):
        provider, mock_client = self._make_provider()
        mock_client.chat.completions.create.return_value = self._make_mock_response()

        provider.generate(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="You are helpful.",
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=None,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "hi"}

    def test_generate_no_system_prompt(self):
        provider, mock_client = self._make_provider()
        mock_client.chat.completions.create.return_value = self._make_mock_response()

        provider.generate(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            system_prompt=None,
            temperature=0.5,
            max_output_tokens=500,
            tool_schemas=None,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_generate_passes_tools(self):
        provider, mock_client = self._make_provider()
        mock_client.chat.completions.create.return_value = self._make_mock_response()

        tools = [{"type": "function", "function": {"name": "test"}}]
        provider.generate(
            model="gpt-4o",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=tools,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == tools

    def test_generate_extra_body(self):
        provider, mock_client = self._make_provider(
            extra_body={"reasoning": {"enabled": True}}
        )
        mock_client.chat.completions.create.return_value = self._make_mock_response()

        provider.generate(
            model="qwen/qwen3.6-plus:free",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=None,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"reasoning": {"enabled": True}}

    def test_parse_text_response(self):
        provider, _ = self._make_provider()
        response = self._make_mock_response(content="Hello world")

        parsed = provider.parse_response(response)
        assert parsed.text == "Hello world"
        assert parsed.tool_calls == []
        assert parsed.usage.prompt_tokens == 100
        assert parsed.usage.completion_tokens == 50

    def test_parse_tool_call_response(self):
        provider, _ = self._make_provider()

        tc = self._make_mock_tool_call("call_123", "search", {"query": "test"})
        response = self._make_mock_response(content=None, tool_calls=[tc])

        parsed = provider.parse_response(response)
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].id == "call_123"
        assert parsed.tool_calls[0].name == "search"
        assert parsed.tool_calls[0].args == {"query": "test"}

    def test_parse_reasoning_details(self):
        provider, _ = self._make_provider(preserve_reasoning=True)

        response = self._make_mock_response(
            content="3 r's",
            reasoning_details=[{"content": "Let me count..."}, {"content": " s-t-r-a-w-b-e-r-r-y"}],
        )

        parsed = provider.parse_response(response)
        assert parsed.thinking_text == "Let me count... s-t-r-a-w-b-e-r-r-y"

    def test_parse_reasoning_disabled(self):
        provider, _ = self._make_provider(preserve_reasoning=False)

        response = self._make_mock_response(
            content="answer",
            reasoning_details=[{"content": "thinking"}],
        )

        parsed = provider.parse_response(response)
        assert parsed.thinking_text is None

    def test_build_user_message(self):
        provider, _ = self._make_provider()
        msg = provider.build_user_message("hello")
        assert msg == {"role": "user", "content": "hello"}

    def test_build_tool_result_messages(self):
        provider, _ = self._make_provider()

        tool_calls = [
            ToolCall(id="call_1", name="search", args={"q": "x"}),
            ToolCall(id="call_2", name="calc", args={"n": 5}),
        ]
        results = [
            ("search", {"results": ["a", "b"]}),
            ("calc", {"answer": 42}),
        ]

        msgs = provider.build_tool_result_messages(tool_calls, results)
        assert isinstance(msgs, list)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_1"
        assert json.loads(msgs[0]["content"]) == {"results": ["a", "b"]}
        assert msgs[1]["tool_call_id"] == "call_2"

    def test_build_tool_schemas(self):
        provider, _ = self._make_provider()

        def my_tool(query: str) -> str:
            """Search for something."""
            return ""

        schemas = provider.build_tool_schemas([my_tool])
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "my_tool"

    def test_build_tool_schemas_empty(self):
        provider, _ = self._make_provider()
        assert provider.build_tool_schemas([]) is None

    def test_serialize_deserialize_roundtrip(self):
        provider, _ = self._make_provider()

        msg = {"role": "user", "content": "hello"}
        serialized = provider.serialize_message(msg)
        assert serialized["_provider"] == "openai"

        deserialized = provider.deserialize_message(serialized)
        assert deserialized == msg  # _provider tag stripped

    def test_format_user_message(self):
        provider, _ = self._make_provider()
        result = provider.format_message_for_display({"role": "user", "content": "hi"})
        assert result == {"role": "user", "content": "hi"}

    def test_format_tool_response(self):
        provider, _ = self._make_provider()
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "result"}
        result = provider.format_message_for_display(msg)
        assert result["role"] == "tool"
        assert "call_1" in result["content"]

    def test_is_retryable_error(self):
        provider, _ = self._make_provider()
        # Generic exception is not retryable
        assert provider.is_retryable_error(ValueError("nope")) is False

    def test_cache_config_ignored(self):
        """OpenAI provider should ignore cache_config without error."""
        provider, mock_client = self._make_provider()
        mock_client.chat.completions.create.return_value = self._make_mock_response()

        # Should not raise
        provider.generate(
            model="gpt-4o",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=None,
            cache_config={"cache_name": "something", "contents_offset": 5},
        )


# ============================================================
# GeminiProvider tests
# ============================================================


class TestGeminiProvider:
    """Test GeminiProvider with mocked genai client."""

    def test_parse_text_response(self):
        from unittest.mock import patch
        from agent_core.providers.gemini import GeminiProvider

        mock_client = MagicMock()
        with patch("agent_core.providers.gemini.genai"):
            provider = GeminiProvider(client=mock_client)

        # Build a mock response
        part = MagicMock()
        part.function_call = None
        part.text = "Hello"
        content = MagicMock()
        content.parts = [part]
        candidate = MagicMock()
        candidate.content = content
        response = MagicMock()
        response.candidates = [candidate]
        response.text = "Hello"
        response.usage_metadata = None

        parsed = provider.parse_response(response)
        assert parsed.text == "Hello"
        assert parsed.tool_calls == []

    def test_parse_tool_call_response(self):
        from unittest.mock import patch
        from agent_core.providers.gemini import GeminiProvider

        mock_client = MagicMock()
        with patch("agent_core.providers.gemini.genai"):
            provider = GeminiProvider(client=mock_client)

        fc = MagicMock()
        fc.name = "search"
        fc.args = {"query": "test"}
        part = MagicMock()
        part.function_call = fc
        part.text = None
        content = MagicMock()
        content.parts = [part]
        candidate = MagicMock()
        candidate.content = content
        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata = None

        parsed = provider.parse_response(response)
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "search"
        assert parsed.tool_calls[0].args == {"query": "test"}

    def test_client_property(self):
        from unittest.mock import patch
        from agent_core.providers.gemini import GeminiProvider

        mock_client = MagicMock()
        with patch("agent_core.providers.gemini.genai"):
            provider = GeminiProvider(client=mock_client)
        assert provider.client is mock_client

    def test_build_user_message(self):
        from unittest.mock import patch
        from agent_core.providers.gemini import GeminiProvider

        mock_client = MagicMock()
        with patch("agent_core.providers.gemini.genai"):
            with patch("agent_core.providers.gemini.types") as mock_types:
                mock_types.Content = MagicMock()
                mock_types.Part = MagicMock()
                provider = GeminiProvider(client=mock_client)
                provider.build_user_message("hello")
                mock_types.Content.assert_called_once()


# ============================================================
# Protocol compliance
# ============================================================


class TestProtocolCompliance:
    """Verify providers satisfy the LLMProvider protocol."""

    def test_gemini_is_llm_provider(self):
        from agent_core.providers.gemini import GeminiProvider
        assert issubclass(GeminiProvider, LLMProvider)

    def test_openai_is_llm_provider(self):
        from agent_core.providers.openai import OpenAIProvider
        assert issubclass(OpenAIProvider, LLMProvider)
