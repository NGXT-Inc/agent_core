"""Tests for conversation persistence implementations."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_core.core.persistence import (
    InMemoryConversationStore,
    SQLiteConversationStore,
    serialize_content,
    deserialize_content,
    serialize_history,
    deserialize_history,
)


# --- Mock types for serialization tests ---


class MockFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class MockFunctionResponse:
    def __init__(self, name, response):
        self.name = name
        self.response = response


class MockPart:
    def __init__(self, text=None, function_call=None, function_response=None, inline_data=None, thought=None):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response
        self.inline_data = inline_data
        self.thought = thought


class MockContent:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts


class TestInMemoryConversationStore:
    """Tests for the in-memory store."""

    def test_load_empty(self):
        store = InMemoryConversationStore()
        result = store.load("session-1", "designer")
        assert result == []

    def test_save_and_load(self):
        store = InMemoryConversationStore()
        history = [MockContent(role="user", parts=[MockPart(text="hello")])]

        store.save("session-1", "designer", history)
        result = store.load("session-1", "designer")

        assert len(result) == 1
        assert result[0].role == "user"

    def test_isolation_between_sessions(self):
        store = InMemoryConversationStore()
        store.save("session-1", "designer", [MockContent(role="user", parts=[])])
        store.save("session-2", "designer", [
            MockContent(role="user", parts=[]),
            MockContent(role="model", parts=[]),
        ])

        assert len(store.load("session-1", "designer")) == 1
        assert len(store.load("session-2", "designer")) == 2

    def test_isolation_between_agent_types(self):
        store = InMemoryConversationStore()
        store.save("session-1", "designer", [MockContent(role="user", parts=[])])
        store.save("session-1", "researcher", [
            MockContent(role="user", parts=[]),
            MockContent(role="model", parts=[]),
        ])

        assert len(store.load("session-1", "designer")) == 1
        assert len(store.load("session-1", "researcher")) == 2

    def test_clear(self):
        store = InMemoryConversationStore()
        store.save("session-1", "designer", [MockContent(role="user", parts=[])])
        store.clear("session-1", "designer")

        assert store.load("session-1", "designer") == []

    def test_clear_nonexistent(self):
        store = InMemoryConversationStore()
        store.clear("nonexistent", "agent")  # Should not raise

    def test_clear_all(self):
        store = InMemoryConversationStore()
        store.save("s1", "a1", [MockContent(role="user", parts=[])])
        store.save("s2", "a2", [MockContent(role="user", parts=[])])

        store.clear_all()

        assert store.load("s1", "a1") == []
        assert store.load("s2", "a2") == []

    def test_save_returns_copy(self):
        """Modifying returned history should not affect stored data."""
        store = InMemoryConversationStore()
        original = [MockContent(role="user", parts=[])]
        store.save("s1", "a1", original)

        loaded = store.load("s1", "a1")
        loaded.append(MockContent(role="model", parts=[]))

        # Original store should be unaffected
        assert len(store.load("s1", "a1")) == 1


class TestSQLiteConversationStore:
    """Tests for the SQLite store."""

    def test_creates_db_file(self, temp_db):
        store = SQLiteConversationStore(temp_db)
        assert temp_db.exists()

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "nested" / "dir" / "test.db"
        store = SQLiteConversationStore(db_path)
        assert db_path.exists()

    def test_load_empty(self, temp_db):
        store = SQLiteConversationStore(temp_db)
        result = store.load("session-1", "designer")
        assert result == []

    def test_clear(self, temp_db):
        store = SQLiteConversationStore(temp_db)

        # We need real genai types for SQLite serialization
        # Use the serialize/deserialize helpers directly
        from google.genai import types

        history = [
            types.Content(role="user", parts=[types.Part.from_text(text="hello")]),
        ]

        store.save("session-1", "designer", history)
        assert len(store.load("session-1", "designer")) == 1

        store.clear("session-1", "designer")
        assert store.load("session-1", "designer") == []

    def test_clear_session(self, temp_db):
        store = SQLiteConversationStore(temp_db)
        from google.genai import types

        history = [types.Content(role="user", parts=[types.Part.from_text(text="hi")])]

        store.save("session-1", "designer", history)
        store.save("session-1", "researcher", history)

        store.clear_session("session-1")

        assert store.load("session-1", "designer") == []
        assert store.load("session-1", "researcher") == []

    def test_list_sessions(self, temp_db):
        store = SQLiteConversationStore(temp_db)
        from google.genai import types

        history = [types.Content(role="user", parts=[types.Part.from_text(text="hi")])]

        store.save("session-1", "agent", history)
        store.save("session-2", "agent", history)

        sessions = store.list_sessions()
        assert set(sessions) == {"session-1", "session-2"}

    def test_upsert_behavior(self, temp_db):
        """Saving to same session/agent should overwrite."""
        store = SQLiteConversationStore(temp_db)
        from google.genai import types

        store.save("s1", "a1", [
            types.Content(role="user", parts=[types.Part.from_text(text="first")]),
        ])
        store.save("s1", "a1", [
            types.Content(role="user", parts=[types.Part.from_text(text="second")]),
            types.Content(role="model", parts=[types.Part.from_text(text="reply")]),
        ])

        loaded = store.load("s1", "a1")
        assert len(loaded) == 2

    def test_openai_dict_messages_roundtrip_without_provider(self, temp_db):
        """OpenAI-style dict histories should not need a provider argument."""
        store = SQLiteConversationStore(temp_db)
        history = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{\"query\": \"x\"}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\": true}"},
        ]

        store.save("s-openai", "agent", history)
        loaded = store.load("s-openai", "agent")

        assert loaded == history


class TestSerialization:
    """Test Content serialization/deserialization helpers."""

    def test_serialize_text_part(self):
        content = MockContent(role="user", parts=[MockPart(text="hello world")])
        result = serialize_content(content)
        assert result["role"] == "user"
        assert result["parts"][0] == {"type": "text", "text": "hello world"}

    def test_serialize_function_call(self):
        fc = MockFunctionCall("search", {"query": "test"})
        content = MockContent(role="model", parts=[MockPart(function_call=fc)])
        result = serialize_content(content)
        assert result["parts"][0]["type"] == "function_call"
        assert result["parts"][0]["name"] == "search"
        assert result["parts"][0]["args"] == {"query": "test"}

    def test_serialize_function_response(self):
        fr = MockFunctionResponse("search", {"results": ["a", "b"]})
        content = MockContent(role="user", parts=[MockPart(function_response=fr)])
        result = serialize_content(content)
        assert result["parts"][0]["type"] == "function_response"
        assert result["parts"][0]["name"] == "search"

    def test_serialize_inline_data_skipped(self):
        inline = MagicMock()
        inline.mime_type = "image/png"
        content = MockContent(role="user", parts=[MockPart(inline_data=inline)])
        result = serialize_content(content)
        assert result["parts"][0]["type"] == "inline_data"
        assert result["parts"][0]["skipped"] is True

    def test_roundtrip_text(self):
        """Serialize then deserialize text content."""
        from google.genai import types

        original = types.Content(
            role="user",
            parts=[types.Part.from_text(text="hello")],
        )
        serialized = serialize_content(original)
        restored = deserialize_content(serialized)

        assert restored.role == "user"
        assert len(restored.parts) == 1

    def test_roundtrip_history(self):
        """Serialize and deserialize a full history list."""
        from google.genai import types

        history = [
            types.Content(role="user", parts=[types.Part.from_text(text="question")]),
            types.Content(role="model", parts=[types.Part.from_text(text="answer")]),
        ]

        json_str = serialize_history(history)
        restored = deserialize_history(json_str)

        assert len(restored) == 2
        assert restored[0].role == "user"
        assert restored[1].role == "model"
