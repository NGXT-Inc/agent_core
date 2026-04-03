"""Persistence interfaces and implementations for agent_core.

This module provides:
1. ConversationStoreProtocol - Interface for conversation persistence
2. InMemoryConversationStore - Simple in-memory implementation (default)
3. SQLiteConversationStore - File-based persistence (optional)

Applications can implement their own stores (Redis, PostgreSQL, etc.)
by following the ConversationStoreProtocol.

The Gemini-specific ``serialize_content`` / ``deserialize_content`` functions
are kept for backward compatibility (used by Papyrus chat_persistence.py).
For provider-agnostic serialization, use ``serialize_message`` / ``deserialize_message``.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from google.genai import types

logger = logging.getLogger(__name__)


@runtime_checkable
class ConversationStoreProtocol(Protocol):
    """Protocol for conversation persistence backends.

    Implement this interface to provide custom persistence
    (e.g., Redis, PostgreSQL, file-based, cloud storage).

    Messages are provider-specific opaque objects (``Any``).
    """

    def load(self, session_id: str, agent_type: str) -> list[Any]:
        """Load conversation history for a session/agent pair.

        Args:
            session_id: Unique session identifier.
            agent_type: Type of agent (e.g., "designer", "researcher").

        Returns:
            List of provider-specific message objects.
        """
        ...

    def save(self, session_id: str, agent_type: str, history: list[Any]) -> None:
        """Save conversation history for a session/agent pair.

        Args:
            session_id: Unique session identifier.
            agent_type: Type of agent.
            history: List of provider-specific message objects.
        """
        ...

    def clear(self, session_id: str, agent_type: str) -> None:
        """Clear conversation history for a session/agent pair.

        Args:
            session_id: Unique session identifier.
            agent_type: Type of agent.
        """
        ...


class InMemoryConversationStore:
    """Simple in-memory conversation store.

    Useful for:
    - Testing
    - Single-session applications
    - Cases where persistence isn't needed

    Note: Data is lost when the process exits.
    """

    def __init__(self):
        self._store: dict[tuple[str, str], list[Any]] = {}

    def load(self, session_id: str, agent_type: str) -> list[Any]:
        key = (session_id, agent_type)
        return self._store.get(key, []).copy()

    def save(self, session_id: str, agent_type: str, history: list[Any]) -> None:
        key = (session_id, agent_type)
        self._store[key] = history.copy()

    def clear(self, session_id: str, agent_type: str) -> None:
        key = (session_id, agent_type)
        self._store.pop(key, None)

    def clear_all(self) -> None:
        """Clear all stored conversations."""
        self._store.clear()


# --- Serialization helpers for SQLite storage ---


def serialize_content(content: types.Content) -> dict[str, Any]:
    """Serialize a Gemini Content object to a JSON-compatible dict."""
    serialized_parts = []

    for part in content.parts:
        if hasattr(part, "text") and part.text is not None:
            serialized_parts.append({"type": "text", "text": part.text})

        elif hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            serialized_parts.append({
                "type": "function_call",
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            })

        elif hasattr(part, "function_response") and part.function_response:
            fr = part.function_response
            serialized_parts.append({
                "type": "function_response",
                "name": fr.name,
                "response": fr.response if isinstance(fr.response, dict) else {"result": str(fr.response)},
            })

        elif hasattr(part, "inline_data") and part.inline_data:
            # Skip binary data (images) - too large for conversation resumption
            serialized_parts.append({
                "type": "inline_data",
                "mime_type": getattr(part.inline_data, "mime_type", "unknown"),
                "skipped": True,
            })

        elif hasattr(part, "thought") and part.thought:
            serialized_parts.append({"type": "thought", "thought": str(part.thought)})

    return {
        "role": content.role,
        "parts": serialized_parts,
    }


def deserialize_content(data: dict[str, Any]) -> types.Content:
    """Deserialize a dict back into a Gemini Content object."""
    parts = []

    for part_data in data.get("parts", []):
        part_type = part_data.get("type")

        if part_type == "text":
            parts.append(types.Part.from_text(text=part_data.get("text", "")))

        elif part_type == "function_call":
            parts.append(types.Part.from_function_call(
                name=part_data.get("name", ""),
                args=part_data.get("args", {}),
            ))

        elif part_type == "function_response":
            parts.append(types.Part.from_function_response(
                name=part_data.get("name", ""),
                response=part_data.get("response", {}),
            ))

        elif part_type == "thought":
            # Restore thought as text (Gemini SDK doesn't have Part.from_thought)
            thought_text = part_data.get("thought", "")
            if thought_text:
                parts.append(types.Part.from_text(text=thought_text))

        elif part_type == "inline_data":
            # Binary data was intentionally skipped during serialization
            pass

        else:
            logger.warning("Unknown part type during deserialization: %s", part_type)

    return types.Content(role=data.get("role", "user"), parts=parts)


def serialize_history(history: list[types.Content]) -> str:
    """Serialize a list of Content objects to JSON string."""
    serialized = [serialize_content(c) for c in history]
    return json.dumps(serialized)


def deserialize_history(json_str: str) -> list[types.Content]:
    """Deserialize a JSON string back to list of Content objects."""
    data = json.loads(json_str)
    return [deserialize_content(d) for d in data]


def serialize_message(message: Any, provider: Any = None) -> dict[str, Any]:
    """Serialize a provider message to a JSON-safe dict.

    If *provider* is given, delegates to ``provider.serialize_message()``.
    Otherwise falls back to Gemini-specific ``serialize_content()``.
    """
    if provider is not None:
        return provider.serialize_message(message)
    return serialize_content(message)


def deserialize_message(data: dict[str, Any], provider: Any = None) -> Any:
    """Deserialize a JSON dict to a provider message.

    If *provider* is given, delegates to ``provider.deserialize_message()``.
    Otherwise falls back to Gemini-specific ``deserialize_content()``.
    """
    if provider is not None:
        return provider.deserialize_message(data)
    return deserialize_content(data)


class SQLiteConversationStore:
    """SQLite-based conversation persistence.

    Stores conversations in a local SQLite database file.
    Suitable for desktop applications and development.

    Args:
        db_path: Path to SQLite database file. Created if doesn't exist.
        provider: Optional LLMProvider for serialization. If ``None``,
            uses Gemini-specific serialization (backward compat).
    """

    def __init__(self, db_path: str | Path, provider: Any = None):
        self.db_path = Path(db_path)
        self._provider = provider
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    history TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, agent_type)
                )
            """)
            conn.commit()

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def load(self, session_id: str, agent_type: str) -> list[Any]:
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT history FROM conversations WHERE session_id = ? AND agent_type = ?",
                (session_id, agent_type),
            )
            row = cursor.fetchone()

        if row and row["history"]:
            data = json.loads(row["history"])
            return [deserialize_message(d, self._provider) for d in data]
        return []

    def save(self, session_id: str, agent_type: str, history: list[Any]) -> None:
        serialized = [serialize_message(m, self._provider) for m in history]
        history_json = json.dumps(serialized)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (session_id, agent_type, history, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (session_id, agent_type)
                DO UPDATE SET history = excluded.history, updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, agent_type, history_json),
            )
            conn.commit()

    def clear(self, session_id: str, agent_type: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE session_id = ? AND agent_type = ?",
                (session_id, agent_type),
            )
            conn.commit()

    def clear_session(self, session_id: str) -> None:
        """Clear all conversations for a session."""
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

    def list_sessions(self) -> list[str]:
        """List all session IDs with stored conversations."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT DISTINCT session_id FROM conversations")
            return [row["session_id"] for row in cursor.fetchall()]
