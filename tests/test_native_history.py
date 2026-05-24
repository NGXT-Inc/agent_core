"""Tests for the C++ ``HistoryStore`` and the Python ``NativeHistory`` wrapper.

These cover the Phase 2 surface: in-process storage, persistence via the
background ``SqliteWriter``, rehydration from disk, and the list-shaped Python
API the agent loop consumes.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_core import _native
from agent_core.core.native_history import NativeHistory


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "history.db")


# ============================================================
# Direct C++ HistoryStore
# ============================================================


class TestRawHistoryStore:
    """Smoke tests for the raw ``_native.HistoryStore`` bindings."""

    def test_empty_store_in_memory(self):
        store = _native.HistoryStore("s", "a", "")
        assert len(store) == 0
        assert store.total_approx_tokens() == 0
        assert store.snapshot_native() == []
        assert store.snapshot_canonical() == []

    def test_append_accumulates_tokens(self):
        store = _native.HistoryStore("s", "a", "")
        store.append("py1", '{"role": "user"}', 5, "user", "openai")
        store.append("py2", '{"role": "assistant"}', 7, "assistant", "openai")

        assert len(store) == 2
        assert store.total_approx_tokens() == 12
        assert store.snapshot_native() == ["py1", "py2"]
        roles = [r for r, _ in store.snapshot_role_and_json()]
        assert roles == ["user", "assistant"]

    def test_replace_prefix_collapses_old_messages(self):
        store = _native.HistoryStore("s", "a", "")
        for i in range(5):
            store.append(f"m{i}", f'{{"i": {i}}}', 2, "user", "openai")

        store.replace_prefix(3, "summary", '{"summary": true}', 8, "user", "openai")

        assert len(store) == 3
        assert store.snapshot_canonical() == [
            '{"summary": true}',
            '{"i": 3}',
            '{"i": 4}',
        ]
        # Tokens: 8 (summary) + 2 + 2 = 12
        assert store.total_approx_tokens() == 12

    def test_clear_drops_messages_and_tokens(self):
        store = _native.HistoryStore("s", "a", "")
        store.append("m", '{}', 5, "user", "openai")
        store.clear()
        assert len(store) == 0
        assert store.total_approx_tokens() == 0

    def test_get_native_out_of_range_raises(self):
        store = _native.HistoryStore("s", "a", "")
        with pytest.raises(IndexError):
            store.get_native(0)


# ============================================================
# Persistence end-to-end through the SqliteWriter
# ============================================================


class TestPersistence:
    def test_persistence_roundtrip(self, tmp_db: str):
        store = _native.HistoryStore("sess-1", "agent-x", tmp_db)
        store.append("py1", '{"role": "user", "i": 1}', 5, "user", "openai")
        store.append("py2", '{"role": "user", "i": 2}', 5, "user", "openai")
        _native.flush_writes()

        # Reopen — canonical state survives; native refs come back as None
        # until the Python wrapper rebuilds them via from_canonical().
        reopened = _native.HistoryStore("sess-1", "agent-x", tmp_db)
        assert reopened.snapshot_canonical() == [
            '{"role": "user", "i": 1}',
            '{"role": "user", "i": 2}',
        ]
        assert reopened.snapshot_native() == [None, None]
        assert reopened.total_approx_tokens() > 0

    def test_clear_removes_persisted_row(self, tmp_db: str):
        store = _native.HistoryStore("sess-2", "agent-x", tmp_db)
        store.append("py1", '{"x": 1}', 3, "user", "openai")
        _native.flush_writes()

        # Confirm the row exists, then clear, then confirm it's gone.
        assert _native.sqlite_load(tmp_db, "sess-2", "agent-x") == ['{"x": 1}']
        store.clear()
        _native.flush_writes()
        assert _native.sqlite_load(tmp_db, "sess-2", "agent-x") == []

    def test_writer_coalesces_rapid_writes(self, tmp_db: str):
        """Many quick appends produce one logical row, not many."""
        store = _native.HistoryStore("sess-coalesce", "agent-x", tmp_db)
        for i in range(20):
            store.append(f"py{i}", f'{{"i": {i}}}', 2, "user", "openai")
        _native.flush_writes()

        # Open a side connection and verify we wrote one row, not 20.
        with sqlite3.connect(tmp_db) as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM conversations "
                "WHERE session_id = ? AND agent_type = ?",
                ("sess-coalesce", "agent-x"),
            ).fetchone()
        assert count == 1

    def test_replace_prefix_persists(self, tmp_db: str):
        store = _native.HistoryStore("sess-3", "agent-x", tmp_db)
        for i in range(4):
            store.append(f"py{i}", f'{{"i": {i}}}', 2, "user", "openai")
        store.replace_prefix(3, "py-sum", '{"summary": true}', 3, "user", "openai")
        _native.flush_writes()

        reopened = _native.HistoryStore("sess-3", "agent-x", tmp_db)
        assert reopened.snapshot_canonical() == [
            '{"summary": true}',
            '{"i": 3}',
        ]

    def test_pending_writes_visible_to_load(self, tmp_db: str):
        """An in-flight pending write should be reflected by sqlite_load."""
        store = _native.HistoryStore("sess-pending", "agent-x", tmp_db)
        store.append("py1", '{"v": 1}', 2, "user", "openai")
        # Don't flush — load() should still see the pending state.
        loaded = _native.sqlite_load(tmp_db, "sess-pending", "agent-x")
        assert loaded == ['{"v": 1}']


# ============================================================
# Python NativeHistory wrapper
# ============================================================


class TestNativeHistoryWrapper:
    """The list-shaped Python wrapper used by the agent loop."""

    def _make_provider(self):
        from agent_core.providers.openai import OpenAIProvider

        return OpenAIProvider(client=MagicMock())

    def test_append_and_len(self):
        provider = self._make_provider()
        history = NativeHistory(provider, "s", "a")

        history.append({"role": "user", "content": "hi"})
        history.append({"role": "assistant", "content": "hello"})

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "hi"}
        assert history[1]["role"] == "assistant"

    def test_iter_yields_provider_native(self):
        provider = self._make_provider()
        history = NativeHistory(provider, "s", "a")
        history.append({"role": "user", "content": "x"})
        history.append({"role": "user", "content": "y"})

        items = list(history)
        assert items == [
            {"role": "user", "content": "x"},
            {"role": "user", "content": "y"},
        ]

    def test_full_slice_replacement_supported(self):
        """The compaction loop does ``contents[:] = [...]`` — must work."""
        provider = self._make_provider()
        history = NativeHistory(provider, "s", "a")
        history.append({"role": "user", "content": "old"})
        history.append({"role": "assistant", "content": "older"})

        replacement = [{"role": "user", "content": "summary"}]
        history[:] = replacement

        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "summary"}

    def test_partial_slice_assignment_rejected(self):
        provider = self._make_provider()
        history = NativeHistory(provider, "s", "a")
        with pytest.raises(TypeError, match="full-slice"):
            history[0:1] = [{"role": "user", "content": "x"}]

    def test_clear(self):
        provider = self._make_provider()
        history = NativeHistory(provider, "s", "a")
        history.append({"role": "user", "content": "x"})
        history.clear()
        assert len(history) == 0
        assert bool(history) is False

    def test_eq_with_list(self):
        provider = self._make_provider()
        history = NativeHistory(provider, "s", "a")
        msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        history.extend(msgs)
        assert history == msgs

    def test_resurrection_rebuilds_native_refs(self, tmp_db: str):
        """After process restart, from_canonical() must repopulate provider_native."""
        provider = self._make_provider()
        history = NativeHistory(provider, "sess-r", "agent-x", tmp_db)
        history.append({"role": "user", "content": "alpha"})
        history.append({"role": "user", "content": "beta"})
        history.flush()

        # Simulate restart: build a fresh NativeHistory; it should rehydrate
        # and rebuild native refs through provider.from_canonical().
        reopened = NativeHistory(provider, "sess-r", "agent-x", tmp_db)
        assert len(reopened) == 2
        natives = list(reopened)
        assert natives[0] == {"role": "user", "content": "alpha"}
        assert natives[1] == {"role": "user", "content": "beta"}


# ============================================================
# Agent integration
# ============================================================


class TestAgentUsesNativeHistory:
    """Confirm the agent path now routes history through the C++ store."""

    def test_agent_with_session_id_uses_native_history(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent
        from tests.conftest import make_text_response

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("hello")

        agent = Agent(session_id="agent-native-1")
        assert isinstance(agent._history, NativeHistory)

        agent.run("greet me")
        # user + model = 2 entries
        assert len(agent._history) == 2
        agent.close()

    def test_agent_without_session_id_uses_plain_list(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        agent = Agent()
        assert isinstance(agent._history, list)
        agent.close()

    def test_sqlite_store_routes_through_cpp_writer(
        self, mock_env, mock_genai, tmp_db: str
    ):
        from agent_core.agents.base import Agent
        from agent_core.core.persistence import SQLiteConversationStore
        from tests.conftest import make_text_response

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        store = SQLiteConversationStore(tmp_db)
        agent = Agent(session_id="sess-cpp", conversation_store=store)
        agent.run("hello")
        _native.flush_writes()

        # The C++ writer should have persisted two canonical entries.
        persisted = _native.sqlite_load(tmp_db, "sess-cpp", "base")
        assert len(persisted) == 2
        agent.close()
