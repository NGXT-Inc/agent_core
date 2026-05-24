"""Tests for the global C++ session registry and the Python ``registry`` proxy.

Covers Phase 3 of the C++ migration:

* Acquire/release lifecycle and refcounting
* Hierarchical session ids (``parent:child:grandchild``)
* Subtree cancellation reaching descendants
* Cross-script visibility — a fresh Python import sees the same registry
* Resurrection from SQLite after eviction
* Agent.spawn() composing child session ids correctly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_core import _native, registry
from agent_core.agents.base import Agent
from tests.conftest import make_text_response, make_tool_call_response


# ============================================================
# Raw registry surface
# ============================================================


class TestRegistryAcquire:
    def test_acquire_creates_session(self):
        handle = _native.registry.acquire("alpha", "agent_x")
        assert handle.session_id == "alpha"
        assert handle.agent_type == "agent_x"
        assert handle.is_cancelled is False
        assert handle.ref_count == 1
        handle.close()

    def test_double_acquire_bumps_refcount(self):
        a = _native.registry.acquire("alpha", "agent_x")
        b = _native.registry.acquire("alpha", "agent_x")
        assert a.ref_count == 2
        a.close()
        assert b.ref_count == 1
        b.close()

    def test_close_is_idempotent(self):
        h = _native.registry.acquire("alpha", "agent_x")
        h.close()
        h.close()  # No raise.

    def test_context_manager_closes_on_exit(self):
        with _native.registry.acquire("alpha", "agent_x") as h:
            assert h.session_id == "alpha"
        # After exit the registry should still show the session resident
        # (no refs but idle TTL keeps it).
        peeked = _native.registry.peek("alpha")
        assert peeked is not None
        assert peeked.ref_count == 0

    def test_peek_returns_none_for_unknown(self):
        assert _native.registry.peek("does-not-exist") is None


class TestHierarchy:
    def test_descendants_via_prefix(self):
        roots = [
            _native.registry.acquire("root", "r"),
            _native.registry.acquire("root:child-1", "c"),
            _native.registry.acquire("root:child-1:grand-1", "g"),
            _native.registry.acquire("root:child-2", "c"),
            _native.registry.acquire("other-root", "r"),
        ]
        try:
            assert registry.descendants("root") == [
                "root:child-1",
                "root:child-1:grand-1",
                "root:child-2",
            ]
            assert registry.descendants("root", include_root=True) == [
                "root",
                "root:child-1",
                "root:child-1:grand-1",
                "root:child-2",
            ]
            # A sibling-prefix should NOT match — ``other-root`` is not under ``root``.
            assert "other-root" not in registry.descendants("root")
        finally:
            for h in roots:
                h.close()

    def test_cancel_subtree_propagates(self):
        root = _native.registry.acquire("root", "r")
        child = _native.registry.acquire("root:child-1", "c")
        grand = _native.registry.acquire("root:child-1:grand", "g")
        sibling = _native.registry.acquire("sibling", "s")

        try:
            flagged = registry.cancel_subtree("root")
            assert flagged == 3  # root + child + grand, NOT sibling
            assert root.is_cancelled is True
            assert child.is_cancelled is True
            assert grand.is_cancelled is True
            assert sibling.is_cancelled is False
        finally:
            for h in (root, child, grand, sibling):
                h.close()

    def test_clear_cancellation_targets_subtree(self):
        root = _native.registry.acquire("root", "r")
        child = _native.registry.acquire("root:child", "c")
        registry.cancel_subtree("root")
        registry.clear_cancellation("root", recursive=True)
        assert root.is_cancelled is False
        assert child.is_cancelled is False
        root.close()
        child.close()


class TestSessionInfo:
    def test_session_info_returns_dict_when_resident(self):
        h = _native.registry.acquire("info-sess", "ag")
        try:
            info = registry.session_info("info-sess")
            assert info is not None
            assert info["session_id"] == "info-sess"
            assert info["agent_type"] == "ag"
            assert info["ref_count"] == 1
            assert info["message_count"] == 0
            assert info["is_cancelled"] is False
        finally:
            h.close()

    def test_session_info_returns_none_for_unknown(self):
        assert registry.session_info("never-existed") is None


class TestIdleEviction:
    def test_reaper_evicts_idle_zero_ref_sessions(self):
        _native.registry.acquire("idle-sess", "ag").close()
        assert "idle-sess" in registry.list_active()

        # Force TTL=0 so the next reap_now sweeps immediately.
        registry.set_idle_ttl_seconds(0)
        try:
            reaped = registry.reap_now()
            assert reaped >= 1
            assert "idle-sess" not in registry.list_active()
        finally:
            registry.set_idle_ttl_seconds(1800)

    def test_reaper_preserves_referenced_sessions(self):
        held = _native.registry.acquire("held", "ag")
        try:
            registry.set_idle_ttl_seconds(0)
            registry.reap_now()
            assert "held" in registry.list_active()
        finally:
            registry.set_idle_ttl_seconds(1800)
            held.close()


# ============================================================
# Cross-script visibility
# ============================================================


class TestCrossScriptVisibility:
    """The C++ registry is process-wide — different Python modules see it."""

    def test_reimport_sees_same_registry(self):
        import importlib

        # Acquire in this module
        h = _native.registry.acquire("shared-sess", "agent")
        try:
            # Re-import the proxy module — the singleton is still there.
            mod = importlib.import_module("agent_core.registry")
            assert "shared-sess" in mod.list_active()
        finally:
            h.close()

    def test_two_modules_share_state(self):
        from agent_core import registry as r1
        from agent_core import registry as r2

        h = _native.registry.acquire("shared-2", "agent")
        try:
            assert "shared-2" in r1.list_active()
            assert "shared-2" in r2.list_active()
        finally:
            h.close()


# ============================================================
# Resurrection from SQLite
# ============================================================


class TestResurrection:
    def test_session_resurrected_from_sqlite_after_eviction(self, tmp_path: Path):
        db = str(tmp_path / "res.db")

        h1 = _native.registry.acquire("resur-1", "ag", db)
        h1.history().append(
            "py", '{"role": "user", "x": 1}', 5, "user", "openai"
        )
        _native.flush_writes()
        h1.close()

        # Evict.
        registry.set_idle_ttl_seconds(0)
        try:
            registry.reap_now()
            assert "resur-1" not in registry.list_active()
        finally:
            registry.set_idle_ttl_seconds(1800)

        # Resurrect — same id, same SQLite path → reload from disk.
        h2 = _native.registry.acquire("resur-1", "ag", db)
        try:
            assert len(h2.history()) == 1
            assert h2.history().snapshot_canonical() == [
                '{"role": "user", "x": 1}'
            ]
        finally:
            h2.close()


# ============================================================
# Agent.spawn() composes hierarchical session ids
# ============================================================


class TestAgentSpawn:
    def test_spawn_composes_child_session_id(self, mock_env, mock_genai):
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        class ParentAgent(Agent):
            name = "designer"

        class ChildAgent(Agent):
            name = "explorer"

        parent = ParentAgent(session_id="root-42")
        child = parent.spawn(ChildAgent, suffix="abcd")

        assert child.session_id == "root-42:explorer-abcd"
        assert child._parent_agent == parent.instance_id
        parent.close()
        child.close()

    def test_spawn_uses_random_suffix_by_default(self, mock_env, mock_genai):
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        class ParentAgent(Agent):
            name = "designer"

        class ChildAgent(Agent):
            name = "explorer"

        parent = ParentAgent(session_id="root-42")
        child = parent.spawn(ChildAgent)

        assert child.session_id.startswith("root-42:explorer-")
        suffix = child.session_id.removeprefix("root-42:explorer-")
        assert len(suffix) == 8 and all(c in "0123456789abcdef" for c in suffix)
        parent.close()
        child.close()

    def test_spawn_requires_parent_session_id(self, mock_env, mock_genai):
        class ParentAgent(Agent):
            name = "designer"

        class ChildAgent(Agent):
            name = "explorer"

        parent = ParentAgent()  # no session_id
        with pytest.raises(RuntimeError, match="session_id"):
            parent.spawn(ChildAgent)
        parent.close()

    def test_spawn_rejects_colon_in_name(self, mock_env, mock_genai):
        class ParentAgent(Agent):
            name = "designer"

        class ChildAgent(Agent):
            name = "explorer"

        parent = ParentAgent(session_id="root-42")
        with pytest.raises(ValueError, match="reserved"):
            parent.spawn(ChildAgent, agent_type="bad:name")
        parent.close()


# ============================================================
# Subtree cancellation reaches the agent loop
# ============================================================


class TestSubtreeCancellationStopsAgent:
    def test_parent_cancel_subtree_stops_child_loop(self, mock_env, mock_genai):
        mock_client = mock_genai.Client.return_value
        # Child agent that would otherwise loop forever calling its tool.
        mock_client.models.generate_content.return_value = make_tool_call_response(
            "stub_tool", {}
        )

        class ParentAgent(Agent):
            name = "designer"

        class ChildAgent(Agent):
            name = "explorer"
            MAX_ITERATIONS = 100

        parent = ParentAgent(session_id="cancel-root")
        child = parent.spawn(ChildAgent, suffix="c1")
        call_count = 0

        def stub_tool() -> str:
            """Tool that uses the registry to cancel the whole subtree."""
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                # Anyone, anywhere, can call cancel_subtree by session id.
                registry.cancel_subtree("cancel-root")
            return "ok"

        child.register_tool(stub_tool)
        result = child.run("go")

        assert result == "[Cancelled]"
        assert call_count < 50
        parent.close()
        child.close()

    def test_agent_cancel_propagates_to_spawned_children(self, mock_env, mock_genai):
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        class ParentAgent(Agent):
            name = "designer"

        class ChildAgent(Agent):
            name = "explorer"

        parent = ParentAgent(session_id="prop-root")
        child = parent.spawn(ChildAgent, suffix="c1")
        grand = child.spawn(ChildAgent, agent_type="grandchild", suffix="g1")

        parent.cancel()
        assert parent._session_handle.is_cancelled
        assert child._session_handle.is_cancelled
        assert grand._session_handle.is_cancelled
        parent.close()
        child.close()
        grand.close()
