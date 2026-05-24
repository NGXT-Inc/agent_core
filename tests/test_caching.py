"""Tests for the Vertex AI cache state machine.

Phase 4 of the C++ migration moved the slot state machine and worker pool
into the native extension. The public ``ContextCacheRegistry`` is now a
thin Python facade that delegates everything; tests work against
``registry.peek_slot(agent_id)`` for inspection and ``registry.seed_slot(...)``
for setting up scenarios that previously poked at a dataclass directly.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent_core.core.caching import (
    CacheAdvice,
    ContextCacheRegistry,
    MIN_TOKEN_GROWTH,
    _CacheSlot,
    _compute_fingerprint,
)
from tests.conftest import (
    MockCachedContent,
    MockContent,
    MockPart,
    MockResponse,
    MockTokenCountResponse,
    MockUsageMetadata,
    make_text_response,
    make_tool_call_response,
)


@pytest.fixture
def fresh_registry():
    """Build a registry with a mock client and tear down cleanly."""
    mock_client = MagicMock()
    registry = ContextCacheRegistry(mock_client, max_workers=2, cache_ttl_seconds=600)
    yield registry, mock_client
    registry.close()


# ============================================================
# ContextCacheRegistry — init / register / unregister
# ============================================================


class TestRegistryInit:
    def test_initial_state(self, fresh_registry):
        registry, _ = fresh_registry
        advice = registry.get_advice("nonexistent", None, None)
        assert advice == CacheAdvice(cache_name=None, contents_offset=0)

    def test_register_creates_slot(self, fresh_registry):
        registry, _ = fresh_registry
        registry.register("agent-1", "test-model", min_token_threshold=8192)
        snap = registry.peek_slot("agent-1")
        assert snap is not None
        assert snap.model_name == "test-model"
        assert snap.min_token_threshold == 8192

    def test_unregister_removes_slot(self, fresh_registry):
        registry, _ = fresh_registry
        registry.register("agent-1", "test-model")
        registry.unregister("agent-1")
        assert registry.peek_slot("agent-1") is None

    def test_unregister_nonexistent_is_noop(self, fresh_registry):
        registry, _ = fresh_registry
        registry.unregister("nonexistent")  # no raise

    def test_re_register_drops_old_cache(self, fresh_registry):
        registry, mock_client = fresh_registry
        registry.register("agent-1", "test-model")
        registry.seed_slot("agent-1", ready_name="cachedContents/old")
        registry.register("agent-1", "test-model")
        # Old ready cache must be queued for deletion in the background.
        deadline = time.monotonic() + 2
        seen = False
        while time.monotonic() < deadline:
            for call in mock_client.caches.delete.call_args_list:
                if call.kwargs.get("name") == "cachedContents/old":
                    seen = True
                    break
            if seen:
                break
            time.sleep(0.02)
        assert seen
        # Slot fresh again.
        snap = registry.peek_slot("agent-1")
        assert snap is not None
        assert not snap.has_ready


# ============================================================
# get_advice — promotion, drift, TTL
# ============================================================


class TestGetAdvice:
    def test_no_cache_returns_empty_advice(self, fresh_registry):
        registry, _ = fresh_registry
        registry.register("agent-1", "test-model")
        advice = registry.get_advice("agent-1", "sys prompt", None)
        assert advice.cache_name is None
        assert advice.contents_offset == 0

    def test_ready_cache_returned_when_valid(self, fresh_registry):
        registry, _ = fresh_registry
        registry.register("agent-1", "test-model")
        fp = _compute_fingerprint("sys", None)
        registry.seed_slot(
            "agent-1",
            ready_name="cachedContents/valid",
            ready_offset=4,
            config_fingerprint=fp,
        )
        advice = registry.get_advice("agent-1", "sys", None)
        assert advice.cache_name == "cachedContents/valid"
        assert advice.contents_offset == 4

    def test_config_drift_invalidates_cache(self, fresh_registry):
        registry, mock_client = fresh_registry
        registry.register("agent-1", "test-model")
        fp = _compute_fingerprint("old prompt", None)
        registry.seed_slot(
            "agent-1",
            ready_name="cachedContents/stale",
            ready_offset=3,
            config_fingerprint=fp,
        )

        advice = registry.get_advice("agent-1", "new prompt", None)
        assert advice.cache_name is None
        # Old cache scheduled for delete.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if any(
                c.kwargs.get("name") == "cachedContents/stale"
                for c in mock_client.caches.delete.call_args_list
            ):
                return
            time.sleep(0.02)
        pytest.fail("expected cachedContents/stale to be queued for deletion")

    def test_auto_promotes_done_pending(self, fresh_registry):
        registry, _ = fresh_registry
        registry.register("agent-1", "test-model")
        fp = _compute_fingerprint("sys", None)
        registry.seed_slot(
            "agent-1",
            pending_done=True,
            pending_cache_name="cachedContents/new",
            pending_through_index=5,
            config_fingerprint=fp,
        )
        advice = registry.get_advice("agent-1", "sys", None)
        assert advice.cache_name == "cachedContents/new"
        assert advice.contents_offset == 5
        # Pending cleared after promotion.
        snap = registry.peek_slot("agent-1")
        assert snap is not None and snap.pending is False

    def test_promotion_deletes_old_ready(self, fresh_registry):
        registry, mock_client = fresh_registry
        registry.register("agent-1", "test-model")
        fp = _compute_fingerprint("sys", None)
        registry.seed_slot(
            "agent-1",
            ready_name="cachedContents/old",
            ready_offset=2,
            pending_done=True,
            pending_cache_name="cachedContents/new",
            pending_through_index=5,
            config_fingerprint=fp,
        )
        advice = registry.get_advice("agent-1", "sys", None)
        assert advice.cache_name == "cachedContents/new"

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if any(
                c.kwargs.get("name") == "cachedContents/old"
                for c in mock_client.caches.delete.call_args_list
            ):
                return
            time.sleep(0.02)
        pytest.fail("old cachedContents/old should be deleted on promotion")


# ============================================================
# notify — threshold, pending, token delta
# ============================================================


def _wait_for_pending(registry, agent_id, *, system="sys", tools=None, timeout=2.0):
    """Block until any pending cache creation completes and gets promoted.

    ``get_advice(wait=True)`` blocks for the pending future (up to *timeout*),
    then promotes inside the lock. After this returns the slot should have
    ``has_ready=True`` if the worker callback returned a non-empty name.
    """
    registry.get_advice(agent_id, system, tools, wait=True, wait_timeout=timeout)


class TestNotify:
    def test_fires_cache_creation_above_threshold(self, fresh_registry):
        registry, mock_client = fresh_registry
        mock_client.caches.create.return_value = MockCachedContent(
            "cachedContents/new"
        )
        registry.register("agent-1", "test-model", min_token_threshold=32_768)
        contents = [MockContent(role="user", parts=[MockPart(text="large")])]

        registry.notify("agent-1", contents, "sys", None, token_count=50_000)
        _wait_for_pending(registry, "agent-1")

        mock_client.caches.create.assert_called_once()
        snap = registry.peek_slot("agent-1")
        assert snap is not None and snap.has_ready
        assert snap.ready_name == "cachedContents/new"

    def test_skips_below_threshold(self, fresh_registry):
        registry, mock_client = fresh_registry
        registry.register("agent-1", "test-model", min_token_threshold=32_768)
        contents = [MockContent(role="user", parts=[MockPart(text="small")])]

        registry.notify("agent-1", contents, "sys", None, token_count=1_000)
        time.sleep(0.1)
        mock_client.caches.create.assert_not_called()

    def test_skips_when_pending(self, fresh_registry):
        """A second ``notify`` while pending must not enqueue another create."""
        registry, mock_client = fresh_registry
        slow_event = threading.Event()

        def slow_create(*args, **kwargs):
            slow_event.wait(timeout=5)
            return MockCachedContent("cachedContents/slow")

        mock_client.caches.create.side_effect = slow_create
        registry.register("agent-1", "test-model", min_token_threshold=100)
        contents = [MockContent(role="user", parts=[MockPart(text="data")])]

        try:
            registry.notify("agent-1", contents, "sys", None, token_count=60_000)
            # Wait until the worker has the job and pending is set.
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                snap = registry.peek_slot("agent-1")
                if snap is not None and snap.pending:
                    break
                time.sleep(0.01)
            # Second notify while pending — must not call create again.
            registry.notify("agent-1", contents, "sys", None, token_count=60_000)
            time.sleep(0.05)
            assert mock_client.caches.create.call_count == 1
        finally:
            slow_event.set()

    def test_token_delta_guard(self, fresh_registry):
        registry, mock_client = fresh_registry
        registry.register("agent-1", "test-model", min_token_threshold=32_768)
        # Slot's last_cache_token_count is 50_000.
        registry.seed_slot(
            "agent-1",
            last_cache_token_count=50_000,
            config_fingerprint=_compute_fingerprint("sys", None),
        )
        contents = [MockContent(role="user", parts=[MockPart(text="data")])]
        # Only +1000 new tokens — below MIN_TOKEN_GROWTH (4096).
        registry.notify("agent-1", contents, "sys", None, token_count=51_000)
        time.sleep(0.05)
        mock_client.caches.create.assert_not_called()

    def test_count_tokens_fallback(self, fresh_registry):
        registry, mock_client = fresh_registry
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(50_000)
        mock_client.caches.create.return_value = MockCachedContent()
        registry.register("agent-1", "test-model", min_token_threshold=32_768)
        contents = [MockContent(role="user", parts=[MockPart(text="data")])]

        registry.notify("agent-1", contents, "sys", None, token_count=None)
        _wait_for_pending(registry, "agent-1")

        mock_client.models.count_tokens.assert_called_once()
        snap = registry.peek_slot("agent-1")
        assert snap is not None and snap.has_ready

    def test_count_tokens_failure_skips(self, fresh_registry):
        registry, mock_client = fresh_registry
        mock_client.models.count_tokens.side_effect = Exception("API Error")
        registry.register("agent-1", "test-model")

        registry.notify("agent-1", [], "sys", None, token_count=None)
        time.sleep(0.05)
        snap = registry.peek_slot("agent-1")
        assert snap is not None and not snap.pending

    def test_notify_for_unregistered_agent_is_noop(self, fresh_registry):
        registry, mock_client = fresh_registry
        registry.notify("nonexistent", [], "sys", None, token_count=50_000)
        time.sleep(0.05)
        mock_client.caches.create.assert_not_called()


# ============================================================
# TTL — expiry returns no advice and queues delete
# ============================================================


class TestTTLExpiry:
    def test_expired_cache_not_returned(self):
        mock_client = MagicMock()
        registry = ContextCacheRegistry(
            mock_client, max_workers=1, cache_ttl_seconds=1
        )
        try:
            registry.register("agent-1", "test-model")
            registry.seed_slot(
                "agent-1",
                ready_name="cachedContents/expired",
                ready_offset=3,
                ready_expired=True,
                config_fingerprint=_compute_fingerprint("sys", None),
            )
            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name is None

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if any(
                    c.kwargs.get("name") == "cachedContents/expired"
                    for c in mock_client.caches.delete.call_args_list
                ):
                    return
                time.sleep(0.02)
            pytest.fail("expired cache should be queued for deletion")
        finally:
            registry.close()

    def test_non_expired_cache_returned(self, fresh_registry):
        registry, _ = fresh_registry
        registry.register("agent-1", "test-model")
        registry.seed_slot(
            "agent-1",
            ready_name="cachedContents/fresh",
            ready_offset=3,
            config_fingerprint=_compute_fingerprint("sys", None),
        )
        advice = registry.get_advice("agent-1", "sys", None)
        assert advice.cache_name == "cachedContents/fresh"


# ============================================================
# invalidate / unregister
# ============================================================


class TestInvalidate:
    def test_clears_ready_cache(self, fresh_registry):
        registry, mock_client = fresh_registry
        registry.register("agent-1", "test-model")
        registry.seed_slot(
            "agent-1", ready_name="cachedContents/active", ready_offset=5
        )

        registry.invalidate("agent-1")

        snap = registry.peek_slot("agent-1")
        assert snap is not None and not snap.has_ready

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if any(
                c.kwargs.get("name") == "cachedContents/active"
                for c in mock_client.caches.delete.call_args_list
            ):
                return
            time.sleep(0.02)
        pytest.fail("ready cache should be queued for deletion")

    def test_invalidate_nonexistent_is_noop(self, fresh_registry):
        registry, _ = fresh_registry
        registry.invalidate("nonexistent")  # no raise

    def test_tolerates_delete_failure(self, fresh_registry):
        registry, mock_client = fresh_registry
        mock_client.caches.delete.side_effect = Exception("Network error")
        registry.register("agent-1", "test-model")
        registry.seed_slot("agent-1", ready_name="cachedContents/stale")

        registry.invalidate("agent-1")  # no raise
        snap = registry.peek_slot("agent-1")
        assert snap is not None and not snap.has_ready


class TestUnregister:
    def test_deletes_ready_cache(self, fresh_registry):
        registry, mock_client = fresh_registry
        registry.register("agent-1", "test-model")
        registry.seed_slot("agent-1", ready_name="cachedContents/active")

        registry.unregister("agent-1")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if any(
                c.kwargs.get("name") == "cachedContents/active"
                for c in mock_client.caches.delete.call_args_list
            ):
                break
            time.sleep(0.02)
        assert registry.peek_slot("agent-1") is None


class TestClose:
    def test_close_is_idempotent(self):
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        registry.close()
        registry.close()  # no raise


# ============================================================
# Multi-agent independence
# ============================================================


class TestMultiAgent:
    def test_independent_slots(self, fresh_registry):
        registry, _ = fresh_registry
        registry.register("agent-1", "model-a")
        registry.register("agent-2", "model-b")
        registry.seed_slot(
            "agent-1",
            ready_name="cachedContents/a1",
            ready_offset=3,
            config_fingerprint=_compute_fingerprint("sys", None),
        )

        advice1 = registry.get_advice("agent-1", "sys", None)
        advice2 = registry.get_advice("agent-2", "sys", None)

        assert advice1.cache_name == "cachedContents/a1"
        assert advice2.cache_name is None

    def test_invalidate_one_doesnt_affect_other(self, fresh_registry):
        registry, _ = fresh_registry
        registry.register("agent-1", "test-model")
        registry.register("agent-2", "test-model")
        fp = _compute_fingerprint("sys", None)
        registry.seed_slot("agent-1", ready_name="cachedContents/a1", config_fingerprint=fp)
        registry.seed_slot("agent-2", ready_name="cachedContents/a2", config_fingerprint=fp)

        registry.invalidate("agent-1")

        snap1 = registry.peek_slot("agent-1")
        snap2 = registry.peek_slot("agent-2")
        assert snap1 is not None and not snap1.has_ready
        assert snap2 is not None and snap2.ready_name == "cachedContents/a2"


# ============================================================
# Fingerprint matches Python's previous hash exactly
# ============================================================


class TestFingerprint:
    def test_fingerprint_is_deterministic(self):
        a = _compute_fingerprint("hello", None)
        b = _compute_fingerprint("hello", None)
        assert a == b
        assert len(a) == 16

    def test_fingerprint_changes_with_system(self):
        a = _compute_fingerprint("hello", None)
        b = _compute_fingerprint("HELLO", None)
        assert a != b

    def test_fingerprint_includes_tool_names(self):
        def tool_a():
            pass

        def tool_b():
            pass

        a = _compute_fingerprint("p", [tool_a])
        b = _compute_fingerprint("p", [tool_a, tool_b])
        assert a != b


# ============================================================
# Agent + Registry integration
# ============================================================


class TestAgentCachingIntegration:
    """Test that Agent.run() correctly uses the cache registry."""

    @pytest.fixture(autouse=True)
    def setup_registry(self, mock_env, mock_genai):
        """Inject a mock ContextCacheRegistry on the Agent class for the test."""
        from agent_core.agents.base import Agent
        from agent_core.core.caching import CacheAdvice

        mock_client = mock_genai.Client.return_value
        mock_registry = MagicMock()
        mock_registry.get_advice.return_value = CacheAdvice(
            cache_name=None, contents_offset=0
        )

        old_registry = Agent._cache_registry
        Agent._cache_registry = mock_registry

        self.mock_client = mock_client
        self.mock_registry = mock_registry
        self.Agent = Agent
        self.CacheAdvice = CacheAdvice
        yield
        Agent._cache_registry = old_registry

    def test_caching_disabled_agent(self):
        class NoCacheAgent(self.Agent):
            ENABLE_CACHING = False

        self.mock_client.models.generate_content.return_value = make_text_response("ok")
        agent = NoCacheAgent()
        assert agent._cache_enabled is False
        self.mock_registry.register.assert_not_called()
        agent.run("test")
        agent.close()

    def test_caching_enabled_registers(self):
        self.mock_client.models.generate_content.return_value = make_text_response("ok")
        agent = self.Agent(session_id="test-session")
        assert agent._cache_enabled is True
        self.mock_registry.register.assert_called_once()
        agent.close()

    def test_openai_provider_does_not_register_gemini_cache(self):
        from agent_core.providers.openai import OpenAIProvider

        provider = OpenAIProvider(client=MagicMock())
        agent = self.Agent(
            provider=provider, model_name="gpt-4o", session_id="test-session"
        )
        assert agent._cache_enabled is False
        self.mock_registry.register.assert_not_called()
        agent.close()

    def test_no_cache_uses_base_config(self):
        self.mock_client.models.generate_content.return_value = make_text_response("ok")
        agent = self.Agent(session_id="test-session")
        agent.run("hello")
        self.mock_registry.get_advice.assert_called()
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        assert len(call_kwargs["contents"]) == 1
        agent.close()

    def test_cache_used_when_available(self):
        self.mock_registry.get_advice.return_value = self.CacheAdvice(
            cache_name="cachedContents/c1", contents_offset=3
        )
        self.mock_client.models.generate_content.return_value = MockResponse(
            "cached response",
            MockContent(role="model", parts=[MockPart(text="cached response")]),
            usage_metadata=MockUsageMetadata(
                prompt_token_count=50_000, cached_content_token_count=45_000
            ),
        )

        agent = self.Agent(session_id="test-session")
        agent._history.extend(
            [
                MockContent(role="user", parts=[MockPart(text="msg1")]),
                MockContent(role="model", parts=[MockPart(text="resp1")]),
                MockContent(role="user", parts=[MockPart(text="msg2")]),
            ]
        )
        result = agent.run("msg3")
        assert result == "cached response"

        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        assert len(call_kwargs["contents"]) == 1
        agent.close()

    def test_notify_called_after_run(self):
        self.mock_client.models.generate_content.return_value = make_text_response("ok")
        agent = self.Agent(session_id="test-session")
        agent.run("hello")
        self.mock_registry.notify.assert_called()
        agent.close()

    def test_cache_fallback_on_failure(self):
        self.mock_registry.get_advice.return_value = self.CacheAdvice(
            cache_name="cachedContents/stale", contents_offset=2
        )
        call_count = [0]

        def generate_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("CachedContent not found")
            return make_text_response("fallback response")

        self.mock_client.models.generate_content.side_effect = generate_side_effect

        agent = self.Agent(session_id="test-session")
        result = agent.run("test prompt")
        assert result == "fallback response"
        self.mock_registry.invalidate.assert_called_with(agent.instance_id)
        agent.close()

    def test_clear_history_invalidates_cache(self):
        self.mock_client.models.generate_content.return_value = make_text_response("ok")
        agent = self.Agent(session_id="test-session")
        agent.clear_history()
        self.mock_registry.invalidate.assert_called_with(agent.instance_id)

    def test_close_unregisters(self):
        self.mock_client.models.generate_content.return_value = make_text_response("ok")
        agent = self.Agent(session_id="test-session")
        instance_id = agent.instance_id
        agent.close()
        self.mock_registry.unregister.assert_called_with(instance_id)
        assert agent._cache_enabled is False

    def test_run_stateless_does_not_notify(self):
        self.mock_client.models.generate_content.return_value = make_text_response(
            "stateless"
        )
        agent = self.Agent(session_id="test-session")
        agent.run_stateless("one-shot query")
        self.mock_registry.notify.assert_not_called()
        agent.close()


class TestContentsOffset:
    def test_offset_zero_sends_full_contents(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")
        agent = Agent()
        assert agent._cache_enabled is False
        agent.run("test")

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert len(call_kwargs["contents"]) == 1
        agent.close()

    def test_offset_skips_cached_prefix(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent
        from agent_core.core.caching import CacheAdvice as PyCacheAdvice

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = MockResponse(
            "ok",
            MockContent(role="model", parts=[MockPart(text="ok")]),
            usage_metadata=MockUsageMetadata(prompt_token_count=50_000),
        )

        mock_registry = MagicMock()
        mock_registry.get_advice.return_value = PyCacheAdvice(
            cache_name="cachedContents/c1", contents_offset=2
        )
        old = Agent._cache_registry
        Agent._cache_registry = mock_registry
        try:
            agent = Agent(session_id="test-session")
            agent._history.extend(
                [
                    MockContent(role="user", parts=[MockPart(text="msg1")]),
                    MockContent(role="model", parts=[MockPart(text="resp1")]),
                ]
            )
            agent.run("msg2")
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            assert len(call_kwargs["contents"]) == 1
            agent.close()
        finally:
            Agent._cache_registry = old


class TestToolCallingWithCache:
    def test_tool_loop_appends_to_full_history(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        responses = [
            make_tool_call_response("my_tool", {"query": "test"}),
            make_text_response("Done!"),
        ]
        mock_client.models.generate_content.side_effect = responses

        agent = Agent()
        assert agent._cache_enabled is False

        def my_tool(query: str) -> str:
            """A test tool."""
            return f"result for {query}"

        agent.register_tool(my_tool)
        result = agent.run("use the tool")
        assert result == "Done!"
        assert len(agent._history) == 4
        agent.close()
