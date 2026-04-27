"""Tests for provider implementations."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from agent_core.providers.types import (
    AgentResponse,
    FilePart,
    FileOutputPart,
    LLMProvider,
    ParsedResponse,
    ProviderCapabilities,
    TextOutputPart,
    TextPart,
    TokenUsage,
    ToolCall,
    UnsupportedInputPart,
    UserMessage,
    coerce_user_message,
)
from tests.conftest import (
    MockContent,
    MockInlineData,
    MockPart,
    MockResponse,
    make_text_response,
)


# ============================================================
# Provider-neutral message part tests
# ============================================================


class TestMessageParts:
    """Test provider-neutral message and attachment helpers."""

    def test_file_part_from_bytes_data_url(self):
        part = FilePart.from_bytes(
            b"hello",
            mime_type="text/plain",
            filename="note.txt",
        )

        assert part.to_base64() == "aGVsbG8="
        assert part.to_data_url() == "data:text/plain;base64,aGVsbG8="
        assert part.placeholder() == (
            "[Attached file omitted after reload: note.txt (text/plain)]"
        )

    def test_file_part_from_path_guesses_mime_type(self, tmp_path: Path):
        file_path = tmp_path / "image.png"
        file_path.write_bytes(b"png")

        part = FilePart.from_path(file_path)

        assert part.data == b"png"
        assert part.filename == "image.png"
        assert part.mime_type == "image/png"
        assert part.is_image is True

    def test_file_part_requires_single_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            FilePart(data=b"x", uri="gs://bucket/file.txt")

    def test_coerce_user_message_adds_attachments(self):
        attachment = FilePart.from_bytes(
            b"data",
            mime_type="application/pdf",
            filename="paper.pdf",
        )

        message = coerce_user_message("Summarize", [attachment])

        assert message.parts == (TextPart("Summarize"), attachment)
        assert message.text == (
            "Summarize\n"
            "[Attached file omitted after reload: paper.pdf (application/pdf)]"
        )

    def test_coerce_user_message_extends_existing_message(self):
        attachment = FilePart.from_uri(
            "gs://bucket/image.jpg",
            mime_type="image/jpeg",
            filename="image.jpg",
        )
        base = UserMessage.from_text("Look at this")

        message = coerce_user_message(base, [attachment])

        assert message.parts == (TextPart("Look at this"), attachment)

    def test_agent_response_text_view(self):
        response = AgentResponse(
            parts=(
                TextOutputPart("hello"),
                FileOutputPart.from_data_url("data:image/png;base64,aW1n"),
            )
        )

        assert response.text == "hello"
        assert str(response) == "hello"
        assert response.parts[1].data == b"img"
        assert response.parts[1].mime_type == "image/png"

    def test_provider_capabilities_match_mime_type_wildcards(self):
        capabilities = ProviderCapabilities(
            supported_input_mime_types=("image/*", "application/pdf")
        )

        assert capabilities.supports_input_mime_type("image/png") is True
        assert capabilities.supports_input_mime_type("application/pdf") is True
        assert capabilities.supports_input_mime_type("text/plain") is False


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

    def test_capabilities_reflect_file_source_support(self):
        provider, _ = self._make_provider()

        capabilities = provider.capabilities("gpt-test")

        assert capabilities.input_images is True
        assert capabilities.input_image_bytes is True
        assert capabilities.input_image_urls is True
        assert capabilities.input_files is True
        assert capabilities.input_file_bytes is True
        assert capabilities.input_file_ids is True
        assert capabilities.input_file_urls is False

    def test_capabilities_allow_non_image_file_urls_when_configured(self):
        provider, _ = self._make_provider(allow_file_urls=True)

        assert provider.capabilities("openrouter-model").input_file_urls is True

    def _make_mock_response(
        self,
        content="Hello",
        tool_calls=None,
        reasoning_details=None,
        prompt_tokens=100,
        completion_tokens=50,
        cached_tokens=0,
        cache_write_tokens=0,
        images=None,
    ):
        """Build a mock OpenAI response."""
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = content
        msg.tool_calls = tool_calls
        msg.reasoning_details = reasoning_details
        msg.images = images

        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        if cached_tokens or cache_write_tokens:
            usage.prompt_tokens_details = {
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
            }
        else:
            usage.prompt_tokens_details = None

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

    def test_generate_applies_openrouter_cache_headers(self):
        provider, mock_client = self._make_provider(
            cache_config={
                "response_cache": True,
                "response_cache_ttl_seconds": 600,
            }
        )
        mock_client.chat.completions.create.return_value = self._make_mock_response()

        provider.generate(
            model="moonshotai/kimi-k2.6",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=None,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_headers"]["X-OpenRouter-Cache"] == "true"
        assert call_kwargs["extra_headers"]["X-OpenRouter-Cache-TTL"] == "600"

    def test_generate_applies_prompt_cache_control(self):
        provider, mock_client = self._make_provider(
            cache_config={"prompt_cache_control": {"type": "ephemeral"}}
        )
        mock_client.chat.completions.create.return_value = self._make_mock_response()

        provider.generate(
            model="anthropic/claude-sonnet-4.6",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=None,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["cache_control"] == {"type": "ephemeral"}

    def test_response_cache_clear_enables_cache_refresh(self):
        provider, mock_client = self._make_provider(
            cache_config={"response_cache": True}
        )
        mock_client.chat.completions.create.return_value = self._make_mock_response()

        provider.generate(
            model="deepseek/deepseek-v4-pro",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=None,
            cache_config={"response_cache": False, "response_cache_clear": True},
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_headers"]["X-OpenRouter-Cache"] == "true"
        assert call_kwargs["extra_headers"]["X-OpenRouter-Cache-Clear"] == "true"

    def test_generate_stream_requests_usage_by_default(self):
        provider, mock_client = self._make_provider()
        mock_client.chat.completions.create.return_value = iter([])

        provider.generate_stream(
            model="gpt-4o",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=None,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream_options"] == {"include_usage": True}

    def test_generate_stream_usage_request_can_be_disabled(self):
        provider, mock_client = self._make_provider(include_stream_usage=False)
        mock_client.chat.completions.create.return_value = iter([])

        provider.generate_stream(
            model="gpt-4o",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=None,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "stream_options" not in call_kwargs

    def test_parse_text_response(self):
        provider, _ = self._make_provider()
        response = self._make_mock_response(content="Hello world")

        parsed = provider.parse_response(response)
        assert parsed.text == "Hello world"
        assert parsed.tool_calls == []
        assert parsed.usage.prompt_tokens == 100
        assert parsed.usage.completion_tokens == 50

    def test_parse_cache_usage_details(self):
        provider, _ = self._make_provider()
        response = self._make_mock_response(
            content="Hello world",
            cached_tokens=80,
            cache_write_tokens=20,
        )

        parsed = provider.parse_response(response)
        assert parsed.usage.cached_tokens == 80
        assert parsed.usage.cache_write_tokens == 20

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

    def test_parse_reasoning_details_supports_openrouter_shape(self):
        provider, _ = self._make_provider(preserve_reasoning=True)

        response = self._make_mock_response(
            content="answer",
            reasoning_details=[
                {"type": "reasoning.text", "text": "Let me think. "},
                {"type": "reasoning.summary", "summary": "Checked the result."},
            ],
        )

        parsed = provider.parse_response(response)
        assert parsed.thinking_text == "Let me think. Checked the result."

    def test_parse_reasoning_disabled(self):
        provider, _ = self._make_provider(preserve_reasoning=False)

        response = self._make_mock_response(
            content="answer",
            reasoning_details=[{"content": "thinking"}],
        )

        parsed = provider.parse_response(response)
        assert parsed.thinking_text is None

    def test_parse_image_output_parts(self):
        provider, _ = self._make_provider()
        response = self._make_mock_response(
            content="Here is the image.",
            images=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aW1n",
                    },
                }
            ],
        )

        parsed = provider.parse_response(response)

        assert parsed.text == "Here is the image."
        assert parsed.output_parts[0].text == "Here is the image."
        assert parsed.output_parts[1].data == b"img"
        assert parsed.output_parts[1].mime_type == "image/png"

    def test_parse_image_output_adds_ephemeral_placeholder_to_history(self):
        provider, _ = self._make_provider()
        response = self._make_mock_response(
            content="Here is the image.",
            images=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aW1n",
                    },
                }
            ],
        )

        parsed = provider.parse_response(response)
        serialized = provider.serialize_message(parsed.raw_message)

        assert parsed.raw_message["content"] == (
            "Here is the image.\n"
            "[Attached file omitted after reload: image attachment (image/png)]"
        )
        assert serialized["content"] == parsed.raw_message["content"]
        assert "aW1n" not in json.dumps(serialized)

    def test_build_user_message(self):
        provider, _ = self._make_provider()
        msg = provider.build_user_message("hello")
        assert msg == {"role": "user", "content": "hello"}

    def test_build_user_message_with_inline_image(self):
        provider, _ = self._make_provider()
        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Describe this",
                [
                    FilePart.from_bytes(
                        b"img",
                        mime_type="image/png",
                        filename="plot.png",
                        detail="low",
                    )
                ],
            )
        )

        assert msg["role"] == "user"
        assert msg["content"][0] == {"type": "text", "text": "Describe this"}
        assert msg["content"][1] == {
            "type": "text",
            "text": "[Attached file: plot.png (image/png)]",
        }
        assert msg["content"][2] == {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,aW1n",
                "detail": "low",
            },
        }

    def test_build_user_message_with_inline_pdf(self):
        provider, _ = self._make_provider()
        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Summarize this",
                [
                    FilePart.from_bytes(
                        b"pdf",
                        mime_type="application/pdf",
                        filename="paper.pdf",
                    )
                ],
            )
        )

        assert msg["content"][2] == {
            "type": "file",
            "file": {
                "filename": "paper.pdf",
                "file_data": "cGRm",
            },
        }

    def test_build_user_message_with_file_id(self):
        provider, _ = self._make_provider()
        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Summarize this",
                [
                    FilePart.from_file_id(
                        "file_123",
                        mime_type="application/pdf",
                        filename="paper.pdf",
                    )
                ],
            )
        )

        assert msg["content"][2] == {
            "type": "file",
            "file": {
                "filename": "paper.pdf",
                "file_id": "file_123",
            },
        }

    def test_build_user_message_rejects_non_image_uri(self):
        provider, _ = self._make_provider()

        with pytest.raises(UnsupportedInputPart, match="file URI"):
            provider.build_user_message(
                UserMessage.from_prompt(
                    "Summarize this",
                    [
                        FilePart.from_uri(
                            "https://example.com/paper.pdf",
                            mime_type="application/pdf",
                            filename="paper.pdf",
                        )
                    ],
                )
            )

    def test_serialize_multimodal_message_replaces_file_with_placeholder(self):
        provider, _ = self._make_provider()
        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Describe this",
                [FilePart.from_bytes(b"img", mime_type="image/png")],
            )
        )

        serialized = provider.serialize_message(msg)
        display = provider.format_message_for_display(serialized)

        assert serialized["content"][2] == {
            "type": "text",
            "text": (
                "[Attached file omitted after reload: image attachment "
                "(image/png)]"
            ),
        }
        assert "aW1n" not in json.dumps(serialized)
        assert "omitted after reload" in display["content"]

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

    def test_build_tool_result_messages_preserves_list_result(self):
        provider, _ = self._make_provider()

        msgs = provider.build_tool_result_messages(
            [ToolCall(id="call_1", name="search", args={})],
            [("search", [{"title": "A"}, {"title": "B"}])],
        )

        assert json.loads(msgs[0]["content"]) == [
            {"title": "A"},
            {"title": "B"},
        ]

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

    def test_compaction_tail_start_moves_before_tool_results(self):
        from agent_core.agents.compaction import select_preserved_tail_start

        provider, _ = self._make_provider()
        messages = [
            {"role": "user", "content": "older"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "first"},
            {"role": "tool", "tool_call_id": "call_2", "content": "second"},
            {"role": "user", "content": "latest"},
        ]

        start = select_preserved_tail_start(
            provider,
            messages,
            tail_token_budget=1,
            min_messages=2,
            max_chars=1000,
        )

        assert start == 1

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

    def test_generate_stream_preserves_reasoning_details(self):
        provider, mock_client = self._make_provider(preserve_reasoning=True)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create.return_value = iter([
            SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role="assistant",
                            reasoning_details=[
                                {
                                    "type": "reasoning.text",
                                    "text": "First thought. ",
                                    "index": 0,
                                }
                            ],
                        )
                    )
                ],
            ),
            SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_details=[
                                SimpleNamespace(
                                    type="reasoning.summary",
                                    summary="Then summary.",
                                    index=1,
                                )
                            ],
                            content="final",
                        )
                    )
                ],
            ),
            SimpleNamespace(usage=usage, choices=[]),
        ])

        response = provider.generate_stream(
            model="openrouter/reasoning-model",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=100,
            tool_schemas=None,
        )
        parsed = provider.parse_response(response)

        assert parsed.text == "final"
        assert parsed.thinking_text == "First thought. Then summary."
        assert parsed.raw_message["reasoning_details"] == [
            {
                "type": "reasoning.text",
                "text": "First thought. ",
                "index": 0,
            },
            {
                "type": "reasoning.summary",
                "summary": "Then summary.",
                "index": 1,
            },
        ]
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.completion_tokens == 5

    def test_generate_stream_preserves_plain_reasoning(self):
        provider, mock_client = self._make_provider(preserve_reasoning=True)
        mock_client.chat.completions.create.return_value = iter([
            SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role="assistant",
                            reasoning="plain ",
                        )
                    )
                ],
            ),
            SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content="reasoning",
                            content="answer",
                        )
                    )
                ],
            ),
        ])

        response = provider.generate_stream(
            model="openrouter/reasoning-model",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=100,
            tool_schemas=None,
        )
        parsed = provider.parse_response(response)

        assert parsed.text == "answer"
        assert parsed.thinking_text == "plain reasoning"
        assert parsed.raw_message["reasoning"] == "plain reasoning"


# ============================================================
# OpenRouterProvider tests
# ============================================================


class TestOpenRouterProvider:
    """Test OpenRouter-specific provider configuration."""

    def test_capabilities_allow_non_image_file_urls(self):
        from agent_core.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(client=MagicMock())

        assert provider.capabilities("google/gemini-2.5-flash").input_file_urls is True

    def test_provider_applies_attribution_and_response_cache_headers(self):
        from agent_core.providers.openrouter import OpenRouterProvider

        mock_client = MagicMock()
        provider = OpenRouterProvider(
            client=mock_client,
            app_url="https://example.com",
            app_name="Agent Core",
            response_cache=True,
            response_cache_ttl_seconds=900,
        )
        mock_client.chat.completions.create.return_value = (
            TestOpenAIProvider()._make_mock_response()
        )

        provider.generate(
            model="moonshotai/kimi-k2.6",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=None,
        )

        headers = mock_client.chat.completions.create.call_args.kwargs["extra_headers"]
        assert headers["HTTP-Referer"] == "https://example.com"
        assert headers["X-Title"] == "Agent Core"
        assert headers["X-OpenRouter-Cache"] == "true"
        assert headers["X-OpenRouter-Cache-TTL"] == "900"

    def test_cache_config_dataclass_maps_to_request_options(self):
        from agent_core.providers.openrouter import (
            OpenRouterCacheConfig,
            OpenRouterProvider,
        )

        mock_client = MagicMock()
        provider = OpenRouterProvider(
            client=mock_client,
            cache_config=OpenRouterCacheConfig(
                response_cache=True,
                response_cache_ttl_seconds=300,
                prompt_cache_control={"type": "ephemeral"},
            ),
        )
        mock_client.chat.completions.create.return_value = (
            TestOpenAIProvider()._make_mock_response()
        )

        provider.generate(
            model="deepseek/deepseek-v4-pro",
            messages=[],
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=1000,
            tool_schemas=None,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_headers"]["X-OpenRouter-Cache"] == "true"
        assert call_kwargs["extra_headers"]["X-OpenRouter-Cache-TTL"] == "300"
        assert call_kwargs["extra_body"]["cache_control"] == {"type": "ephemeral"}

    def test_build_user_message_accepts_pdf_url(self):
        from agent_core.providers.openrouter import OpenRouterProvider

        mock_client = MagicMock()
        provider = OpenRouterProvider(client=mock_client)

        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Summarize this",
                [
                    FilePart.from_uri(
                        "https://example.com/paper.pdf",
                        mime_type="application/pdf",
                        filename="paper.pdf",
                    )
                ],
            )
        )

        assert msg["content"][2] == {
            "type": "file",
            "file": {
                "filename": "paper.pdf",
                "file_data": "https://example.com/paper.pdf",
            },
        }

    def test_build_user_message_uses_data_url_for_inline_pdf(self):
        from agent_core.providers.openrouter import OpenRouterProvider

        mock_client = MagicMock()
        provider = OpenRouterProvider(client=mock_client)

        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Summarize this",
                [
                    FilePart.from_bytes(
                        b"pdf",
                        mime_type="application/pdf",
                        filename="paper.pdf",
                    )
                ],
            )
        )

        assert msg["content"][2]["file"]["file_data"] == (
            "data:application/pdf;base64,cGRm"
        )

    def test_provider_accepts_underscored_env_alias(self):
        from agent_core.providers.openrouter import OpenRouterProvider

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "",
                "OPEN_ROUTER_API_KEY": "alias-key",
            },
            clear=False,
        ):
            with patch("agent_core.providers.openrouter.load_dotenv"):
                with patch("openai.OpenAI") as mock_openai:
                    OpenRouterProvider()

        assert mock_openai.call_args.kwargs["api_key"] == "alias-key"


# ============================================================
# GeminiProvider tests
# ============================================================


class TestGeminiProvider:
    """Test GeminiProvider with mocked genai client."""

    def test_capabilities_support_inline_and_uri_files(self):
        from agent_core.providers.gemini import GeminiProvider

        provider = GeminiProvider(client=MagicMock())

        capabilities = provider.capabilities("gemini-test")
        assert capabilities.input_images is True
        assert capabilities.input_image_bytes is True
        assert capabilities.input_image_urls is True
        assert capabilities.input_files is True
        assert capabilities.input_file_bytes is True
        assert capabilities.input_file_urls is True
        assert capabilities.input_file_ids is False

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

    def test_parse_rich_response_preserves_output_part_order(self):
        from unittest.mock import patch
        from agent_core.providers.gemini import GeminiProvider

        mock_client = MagicMock()
        with patch("agent_core.providers.gemini.genai"):
            provider = GeminiProvider(client=mock_client)

        content = MockContent(
            role="model",
            parts=[
                MockPart(text="Before"),
                MockPart(inline_data=MockInlineData(b"img", "image/png")),
                MockPart(text="After"),
            ],
        )
        response = MockResponse(text="Before\nAfter", content=content)

        parsed = provider.parse_response(response)

        assert parsed.text == "Before\nAfter"
        assert isinstance(parsed.output_parts[0], TextOutputPart)
        assert parsed.output_parts[0].text == "Before"
        assert isinstance(parsed.output_parts[1], FileOutputPart)
        assert parsed.output_parts[1].data == b"img"
        assert isinstance(parsed.output_parts[2], TextOutputPart)
        assert parsed.output_parts[2].text == "After"

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

    def test_build_user_message_with_inline_file(self, mock_genai):
        from agent_core.providers.gemini import GeminiProvider

        mock_client = mock_genai.Client.return_value
        provider = GeminiProvider(client=mock_client)

        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Describe this",
                [
                    FilePart.from_bytes(
                        b"image",
                        mime_type="image/png",
                        filename="plot.png",
                    )
                ],
            )
        )

        assert msg.parts[0].text == "Describe this"
        assert msg.parts[1].text == "[Attached file: plot.png (image/png)]"
        assert msg.parts[2].inline_data.data == b"image"
        assert msg.parts[2].inline_data.mime_type == "image/png"

    def test_build_user_message_with_uri_file(self, mock_genai):
        from agent_core.providers.gemini import GeminiProvider

        mock_client = mock_genai.Client.return_value
        provider = GeminiProvider(client=mock_client)

        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Read this",
                [
                    FilePart.from_uri(
                        "gs://bucket/paper.pdf",
                        mime_type="application/pdf",
                        filename="paper.pdf",
                    )
                ],
            )
        )

        assert msg.parts[1].text == "[Attached file: paper.pdf (application/pdf)]"
        assert msg.parts[2].file_data.file_uri == "gs://bucket/paper.pdf"
        assert msg.parts[2].file_data.mime_type == "application/pdf"

    def test_build_user_message_rejects_provider_file_id(self, mock_genai):
        from agent_core.providers.gemini import GeminiProvider
        from agent_core.providers.types import UnsupportedInputPart

        mock_client = mock_genai.Client.return_value
        provider = GeminiProvider(client=mock_client)

        with pytest.raises(UnsupportedInputPart, match="file IDs"):
            provider.build_user_message(
                UserMessage.from_prompt(
                    "Read this",
                    [FilePart.from_file_id("file_123", filename="paper.pdf")],
                )
            )

    def test_build_tool_result_messages_preserves_regular_files_list(self, mock_genai):
        from agent_core.providers.gemini import GeminiProvider

        mock_client = mock_genai.Client.return_value
        provider = GeminiProvider(client=mock_client)

        msg = provider.build_tool_result_messages(
            [ToolCall(id="call_1", name="list_files", args={})],
            [("list_files", {"files": ["a.py", "b.py"], "count": 2})],
        )

        response = msg.parts[0].function_response.response
        assert response == {"files": ["a.py", "b.py"], "count": 2}

    def test_build_tool_result_messages_preserves_list_result(self, mock_genai):
        from agent_core.providers.gemini import GeminiProvider

        mock_client = mock_genai.Client.return_value
        provider = GeminiProvider(client=mock_client)

        msg = provider.build_tool_result_messages(
            [ToolCall(id="call_1", name="search", args={})],
            [("search", [{"title": "A"}, {"title": "B"}])],
        )

        response = msg.parts[0].function_response.response
        assert response == {"result": [{"title": "A"}, {"title": "B"}]}

    def test_build_tool_result_messages_attaches_file_parts(self, mock_genai):
        from agent_core.providers.gemini import GeminiProvider

        mock_client = mock_genai.Client.return_value
        provider = GeminiProvider(client=mock_client)

        msg = provider.build_tool_result_messages(
            [ToolCall(id="call_1", name="make_plot", args={})],
            [
                (
                    "make_plot",
                    {
                        "summary": "created",
                        "images": [
                            {
                                "data": b"png",
                                "mime_type": "image/png",
                                "filename": "plot.png",
                            }
                        ],
                    },
                )
            ],
        )

        assert msg.parts[0].function_response.response == {"summary": "created"}
        assert msg.parts[1].text == "[Attached file: plot.png (image/png)]"
        assert msg.parts[2].inline_data.data == b"png"

    def test_gemini_serializes_inline_file_as_ephemeral_placeholder(self, mock_genai):
        from agent_core.providers.gemini import GeminiProvider

        mock_client = mock_genai.Client.return_value
        provider = GeminiProvider(client=mock_client)
        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Describe",
                [FilePart.from_bytes(b"png", mime_type="image/png")],
            )
        )

        serialized = provider.serialize_message(msg)
        restored = provider.deserialize_message(serialized)

        assert serialized["parts"][2]["type"] == "ephemeral_file"
        assert restored.parts[2].text == (
            "[Attached file omitted after reload: unnamed attachment (image/png)]"
        )

    def test_generate_stream_aggregates_text_chunks(self, mock_genai):
        from agent_core.providers.gemini import GeminiProvider

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content_stream.return_value = iter([
            make_text_response("stream"),
            make_text_response("ed"),
        ])
        provider = GeminiProvider(client=mock_client)

        deltas = []
        response = provider.generate_stream(
            model="gemini-test",
            messages=[],
            system_prompt="system",
            temperature=0.7,
            max_output_tokens=100,
            tool_schemas=None,
            on_text_delta=deltas.append,
        )
        parsed = provider.parse_response(response)

        assert deltas == ["stream", "ed"]
        assert parsed.text == "streamed"
        assert parsed.streamed_text is True


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

    def test_openrouter_is_llm_provider(self):
        from agent_core.providers.openrouter import OpenRouterProvider
        assert issubclass(OpenRouterProvider, LLMProvider)
