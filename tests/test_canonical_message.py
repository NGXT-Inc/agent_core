"""Roundtrip tests for ``CanonicalMessage`` across every provider.

The canonical form is the wire shape that crosses the Python↔C++ boundary
once the registry / history store land in the native extension. These tests
lock in the contract before any C++ depends on it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent_core.providers.types import (
    CanonicalMessage,
    FilePart,
    UserMessage,
    UserMessageInput,
)


# ============================================================
# Gemini provider
# ============================================================


class TestGeminiCanonical:
    def _make_provider(self, mock_genai):
        from agent_core.providers.gemini import GeminiProvider

        return GeminiProvider(client=mock_genai.Client.return_value)

    def test_user_text_roundtrip(self, mock_genai):
        provider = self._make_provider(mock_genai)
        msg = provider.build_user_message("hello world")

        canonical = provider.to_canonical(msg)

        assert canonical.role == "user"
        assert canonical.provider_tag == "gemini"
        assert canonical.approx_tokens >= 1
        assert "hello world" in canonical.canonical_json
        assert canonical.provider_native is msg

        # In-process: returns the same native ref untouched.
        restored = provider.from_canonical(canonical)
        assert restored is msg

    def test_assistant_role_normalized(self, mock_genai):
        """Gemini's ``role="model"`` becomes canonical ``role="assistant"``."""
        from tests.conftest import MockContent, MockPart

        provider = self._make_provider(mock_genai)
        msg = MockContent(role="model", parts=[MockPart(text="I am the model.")])

        canonical = provider.to_canonical(msg)

        assert canonical.role == "assistant"

    def test_tool_result_role_detected(self, mock_genai):
        """Function-response Parts in a user-role Content surface as ``"tool"``."""
        from tests.conftest import (
            MockContent,
            MockFunctionResponse,
            MockPart,
        )

        provider = self._make_provider(mock_genai)
        msg = MockContent(
            role="user",
            parts=[
                MockPart(function_response=MockFunctionResponse("search", {"ok": 1})),
            ],
        )

        canonical = provider.to_canonical(msg)

        assert canonical.role == "tool"

    def test_canonical_json_drops_attachment_bytes(self, mock_genai):
        provider = self._make_provider(mock_genai)
        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Look",
                [FilePart.from_bytes(b"png-bytes-here", mime_type="image/png")],
            )
        )

        canonical = provider.to_canonical(msg)

        # The PNG bytes never reach the canonical JSON.
        assert b"png-bytes-here".decode() not in canonical.canonical_json
        assert "ephemeral_file" in canonical.canonical_json

    def test_from_canonical_rebuilds_when_native_missing(self, mock_genai):
        """After a restart the provider_native field is empty — JSON is the truth."""
        provider = self._make_provider(mock_genai)
        msg = provider.build_user_message("persisted hello")
        canonical = provider.to_canonical(msg)

        stripped = CanonicalMessage(
            role=canonical.role,
            provider_tag=canonical.provider_tag,
            canonical_json=canonical.canonical_json,
            approx_tokens=canonical.approx_tokens,
            provider_native=None,
        )

        rebuilt = provider.from_canonical(stripped)
        # New object, not the same identity, but same content visible to the model.
        assert rebuilt is not msg
        rendered = provider.format_message_for_display(rebuilt)
        assert rendered is not None
        assert "persisted hello" in rendered["content"]

    def test_from_canonical_rejects_foreign_provider(self, mock_genai):
        provider = self._make_provider(mock_genai)
        foreign = CanonicalMessage(
            role="user",
            provider_tag="openai",
            canonical_json='{"role": "user"}',
            approx_tokens=1,
            provider_native=None,
        )

        with pytest.raises(ValueError, match="provider 'openai'"):
            provider.from_canonical(foreign)


# ============================================================
# OpenAI provider
# ============================================================


class TestOpenAICanonical:
    def _make_provider(self):
        from agent_core.providers.openai import OpenAIProvider

        return OpenAIProvider(client=MagicMock())

    def test_user_text_roundtrip(self):
        provider = self._make_provider()
        msg = provider.build_user_message("hello world")

        canonical = provider.to_canonical(msg)

        assert canonical.role == "user"
        assert canonical.provider_tag == "openai"
        assert canonical.approx_tokens >= 1
        assert "hello world" in canonical.canonical_json
        assert canonical.provider_native is msg

        restored = provider.from_canonical(canonical)
        assert restored == msg

    def test_assistant_message_with_tool_calls(self):
        provider = self._make_provider()
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "x"}'},
                }
            ],
        }

        canonical = provider.to_canonical(msg)

        assert canonical.role == "assistant"
        assert "call_1" in canonical.canonical_json

    def test_tool_role_preserved(self):
        provider = self._make_provider()
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "{}"}

        canonical = provider.to_canonical(msg)

        assert canonical.role == "tool"

    def test_system_role_preserved(self):
        provider = self._make_provider()
        canonical = provider.to_canonical({"role": "system", "content": "You are…"})
        assert canonical.role == "system"

    def test_unknown_role_falls_back_to_user(self):
        provider = self._make_provider()
        canonical = provider.to_canonical({"role": "weirdrole", "content": "hi"})
        assert canonical.role == "user"

    def test_multimodal_message_drops_bytes(self):
        provider = self._make_provider()
        msg = provider.build_user_message(
            UserMessage.from_prompt(
                "Describe",
                [FilePart.from_bytes(b"img-bytes", mime_type="image/png")],
            )
        )

        canonical = provider.to_canonical(msg)

        # Bytes are stripped and replaced with text placeholders by serialize_message.
        assert "img-bytes" not in canonical.canonical_json
        assert "omitted after reload" in canonical.canonical_json

    def test_from_canonical_rebuilds_when_native_missing(self):
        provider = self._make_provider()
        msg = {"role": "user", "content": "persisted hello"}
        canonical = provider.to_canonical(msg)

        stripped = CanonicalMessage(
            role=canonical.role,
            provider_tag=canonical.provider_tag,
            canonical_json=canonical.canonical_json,
            approx_tokens=canonical.approx_tokens,
            provider_native=None,
        )

        rebuilt = provider.from_canonical(stripped)
        assert rebuilt == msg

    def test_from_canonical_rejects_foreign_provider(self):
        provider = self._make_provider()
        foreign = CanonicalMessage(
            role="user",
            provider_tag="gemini",
            canonical_json='{"role": "user", "parts": []}',
            approx_tokens=1,
            provider_native=None,
        )

        with pytest.raises(ValueError, match="provider 'gemini'"):
            provider.from_canonical(foreign)


# ============================================================
# OpenRouter (subclass of OpenAIProvider)
# ============================================================


class TestOpenRouterCanonical:
    def test_inherits_canonical_methods(self):
        from agent_core.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(client=MagicMock())
        msg = {"role": "user", "content": "hi from openrouter"}

        canonical = provider.to_canonical(msg)

        assert canonical.provider_tag == "openai"  # shared serialization tag
        assert canonical.role == "user"
        assert canonical.approx_tokens >= 1


# ============================================================
# approx_tokens
# ============================================================


class TestApproxTokens:
    def test_zero_for_empty(self):
        from agent_core.providers.gemini import GeminiProvider
        from agent_core.providers.openai import OpenAIProvider

        assert GeminiProvider.approx_tokens("") == 0
        assert OpenAIProvider.approx_tokens("") == 0

    def test_quarter_of_length(self):
        from agent_core.providers.openai import OpenAIProvider

        assert OpenAIProvider.approx_tokens("a") == 1
        assert OpenAIProvider.approx_tokens("aaaa") == 1
        assert OpenAIProvider.approx_tokens("a" * 8) == 2
        assert OpenAIProvider.approx_tokens("a" * 9) == 3


# ============================================================
# Schema / contract
# ============================================================


class TestCanonicalSchema:
    def test_canonical_message_is_frozen(self):
        msg = CanonicalMessage(
            role="user",
            provider_tag="openai",
            canonical_json="{}",
            approx_tokens=1,
            provider_native=None,
        )

        with pytest.raises(AttributeError):
            msg.role = "assistant"  # type: ignore[misc]

    def test_canonical_json_is_deterministic(self):
        """sort_keys ensures the same logical message hashes the same way."""
        from agent_core.providers.openai import OpenAIProvider

        provider = OpenAIProvider(client=MagicMock())
        # Two dicts with reversed key order should hash to the same canonical_json.
        a = {"role": "user", "content": "hi"}
        b = {"content": "hi", "role": "user"}

        ca = provider.to_canonical(a)
        cb = provider.to_canonical(b)

        assert ca.canonical_json == cb.canonical_json
