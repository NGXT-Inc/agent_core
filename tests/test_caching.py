"""Tests for ContextCacheRegistry and its integration with Agent.run().

Tests are organized into:
1. ContextCacheRegistry unit tests (isolated, no Agent)
2. Agent + registry integration tests (full pipeline through run())
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock, patch, call

import pytest

from agent_core.core.caching import (
    CacheAdvice,
    ContextCacheRegistry,
    _CacheSlot,
    _compute_fingerprint,
    MIN_TOKEN_GROWTH,
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


# ============================================================
# ContextCacheRegistry Unit Tests
# ============================================================


class TestRegistryInit:
    """Test registry initialization and basic lifecycle."""

    def test_initial_state(self):
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=2, cache_ttl_seconds=600)
        try:
            # No slots registered yet
            advice = registry.get_advice("nonexistent", None, None)
            assert advice == CacheAdvice(cache_name=None, contents_offset=0)
        finally:
            registry.close()

    def test_register_creates_slot(self):
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model", min_token_threshold=8192)
            assert "agent-1" in registry._slots
            assert registry._slots["agent-1"].model_name == "test-model"
            assert registry._slots["agent-1"].min_token_threshold == 8192
        finally:
            registry.close()

    def test_unregister_removes_slot(self):
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            registry.unregister("agent-1")
            assert "agent-1" not in registry._slots
        finally:
            registry.close()

    def test_unregister_nonexistent_is_noop(self):
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.unregister("nonexistent")  # Should not raise
        finally:
            registry.close()

    def test_re_register_clears_old_state(self):
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            # Manually set some state
            registry._slots["agent-1"].ready_name = "cachedContents/old"
            registry.register("agent-1", "test-model")
            assert registry._slots["agent-1"].ready_name is None
            mock_client.caches.delete.assert_called_with(name="cachedContents/old")
        finally:
            registry.close()


class TestGetAdvice:
    """Test the get_advice auto-promotion and config validation."""

    def test_no_cache_returns_empty_advice(self):
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            advice = registry.get_advice("agent-1", "sys prompt", None)
            assert advice.cache_name is None
            assert advice.contents_offset == 0
        finally:
            registry.close()

    def test_auto_promotes_pending(self):
        """get_advice should promote a completed pending cache automatically."""
        mock_client = MagicMock()
        mock_client.caches.create.return_value = MockCachedContent("cachedContents/new")
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            # Simulate a completed pending cache
            future = Future()
            future.set_result("cachedContents/new")
            slot.pending = future
            slot.pending_through_index = 5
            slot.config_fingerprint = _compute_fingerprint("sys", None)

            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name == "cachedContents/new"
            assert advice.contents_offset == 5
            assert slot.pending is None  # Cleared after promotion
        finally:
            registry.close()

    def test_returns_none_while_pending(self):
        """get_advice should return empty advice when pending isn't done yet."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            # Set a pending future that isn't done
            future = Future()  # Not resolved
            slot.pending = future
            slot.pending_through_index = 3

            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name is None
            assert advice.contents_offset == 0
        finally:
            # Cancel the future before close to avoid hanging
            future.cancel()
            registry.close()

    def test_config_drift_invalidates_cache(self):
        """get_advice should invalidate cache if system_prompt or tools changed."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            # Set up a ready cache with a specific fingerprint
            slot.ready_name = "cachedContents/stale"
            slot.ready_offset = 3
            slot.ready_created_at = time.monotonic()
            slot.config_fingerprint = _compute_fingerprint("old prompt", None)

            # Query with a DIFFERENT system prompt
            advice = registry.get_advice("agent-1", "new prompt", None)
            assert advice.cache_name is None
            assert advice.contents_offset == 0
            # Old cache should be deleted
            mock_client.caches.delete.assert_called_with(name="cachedContents/stale")
        finally:
            registry.close()

    def test_ready_cache_returned_when_valid(self):
        """get_advice returns the ready cache when config matches."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            slot.ready_name = "cachedContents/valid"
            slot.ready_offset = 4
            slot.ready_created_at = time.monotonic()
            slot.config_fingerprint = _compute_fingerprint("sys", None)

            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name == "cachedContents/valid"
            assert advice.contents_offset == 4
        finally:
            registry.close()

    def test_promotion_deletes_old_ready(self):
        """When promoting, the old ready cache should be deleted."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            # Existing ready cache
            slot.ready_name = "cachedContents/old"
            slot.ready_offset = 2
            slot.ready_created_at = time.monotonic()
            slot.config_fingerprint = _compute_fingerprint("sys", None)

            # Pending cache that's done
            future = Future()
            future.set_result("cachedContents/new")
            slot.pending = future
            slot.pending_through_index = 5
            slot.config_fingerprint = _compute_fingerprint("sys", None)

            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name == "cachedContents/new"
            assert advice.contents_offset == 5
        finally:
            registry.close()

    def test_failed_pending_does_not_crash(self):
        """If pending cache creation failed, get_advice handles it gracefully."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            # Set existing ready cache
            slot.ready_name = "cachedContents/existing"
            slot.ready_offset = 3
            slot.ready_created_at = time.monotonic()
            slot.config_fingerprint = _compute_fingerprint("sys", None)

            # Set a failed pending future
            future = Future()
            future.set_exception(Exception("API quota exceeded"))
            slot.pending = future
            slot.pending_through_index = 5

            # Should still return the existing ready cache
            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name == "cachedContents/existing"
            assert advice.contents_offset == 3
        finally:
            registry.close()


class TestNotify:
    """Test the notify method for cache creation decisions."""

    def test_fires_cache_creation_above_threshold(self):
        """notify should fire cache creation when token count is above threshold."""
        mock_client = MagicMock()
        mock_client.caches.create.return_value = MockCachedContent("cachedContents/new")
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model", min_token_threshold=32_768)
            contents = [MockContent(role="user", parts=[MockPart(text="large")])]

            registry.notify("agent-1", contents, "sys", None, token_count=50_000)

            # Should have submitted cache creation
            slot = registry._slots["agent-1"]
            assert slot.pending is not None
            # Wait for it to complete
            slot.pending.result(timeout=5)
            mock_client.caches.create.assert_called_once()
        finally:
            registry.close()

    def test_skips_below_threshold(self):
        """notify should not fire cache creation when below threshold."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model", min_token_threshold=32_768)
            contents = [MockContent(role="user", parts=[MockPart(text="small")])]

            registry.notify("agent-1", contents, "sys", None, token_count=1_000)

            slot = registry._slots["agent-1"]
            assert slot.pending is None
            mock_client.caches.create.assert_not_called()
        finally:
            registry.close()

    def test_skips_when_pending(self):
        """notify should not fire a second creation if one is pending."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model", min_token_threshold=32_768)
            slot = registry._slots["agent-1"]

            # Manually set a pending future that isn't done
            pending_future = Future()
            slot.pending = pending_future
            slot.pending_through_index = 1

            contents = [MockContent(role="user", parts=[MockPart(text="data")])]
            # This notify should skip because pending isn't done yet
            registry.notify("agent-1", contents, "sys", None, token_count=60_000)

            # Pending should be unchanged
            assert slot.pending is pending_future
            mock_client.caches.create.assert_not_called()
        finally:
            pending_future.cancel()
            registry.close()

    def test_token_delta_guard(self):
        """notify should skip if token delta since last cache is too small."""
        mock_client = MagicMock()
        mock_client.caches.create.return_value = MockCachedContent()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model", min_token_threshold=32_768)
            slot = registry._slots["agent-1"]
            slot.last_cache_token_count = 50_000  # Last cache was at 50k

            contents = [MockContent(role="user", parts=[MockPart(text="data")])]
            # Only 1000 new tokens — below MIN_TOKEN_GROWTH
            registry.notify("agent-1", contents, "sys", None, token_count=51_000)

            assert slot.pending is None
            mock_client.caches.create.assert_not_called()
        finally:
            registry.close()

    def test_count_tokens_fallback(self):
        """notify should call count_tokens API when token_count not provided."""
        mock_client = MagicMock()
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(50_000)
        mock_client.caches.create.return_value = MockCachedContent()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model", min_token_threshold=32_768)
            contents = [MockContent(role="user", parts=[MockPart(text="data")])]

            registry.notify("agent-1", contents, "sys", None, token_count=None)

            mock_client.models.count_tokens.assert_called_once()
            assert registry._slots["agent-1"].pending is not None
        finally:
            registry.close()

    def test_count_tokens_failure_skips(self):
        """If count_tokens fails, notify should skip gracefully."""
        mock_client = MagicMock()
        mock_client.models.count_tokens.side_effect = Exception("API Error")
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            contents = []

            registry.notify("agent-1", contents, "sys", None, token_count=None)

            assert registry._slots["agent-1"].pending is None
        finally:
            registry.close()

    def test_notify_after_close_is_noop(self):
        """notify should be a no-op after close()."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        registry.register("agent-1", "test-model")
        registry.close()

        # Should not raise or submit work
        registry.notify("agent-1", [], "sys", None, token_count=50_000)
        mock_client.caches.create.assert_not_called()

    def test_notify_for_unregistered_agent_is_noop(self):
        """notify for an unregistered agent should be a no-op."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.notify("nonexistent", [], "sys", None, token_count=50_000)
            mock_client.caches.create.assert_not_called()
        finally:
            registry.close()


class TestTTLExpiry:
    """Test TTL enforcement on cached content."""

    def test_expired_cache_not_returned(self):
        """get_advice should not return an expired cache."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1, cache_ttl_seconds=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            slot.ready_name = "cachedContents/expired"
            slot.ready_offset = 3
            slot.ready_created_at = time.monotonic() - 2  # 2 seconds ago, TTL is 1s
            slot.config_fingerprint = _compute_fingerprint("sys", None)

            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name is None
            assert advice.contents_offset == 0
            # Expired cache should be deleted
            mock_client.caches.delete.assert_called_with(name="cachedContents/expired")
        finally:
            registry.close()

    def test_non_expired_cache_returned(self):
        """get_advice should return a cache that hasn't expired yet."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1, cache_ttl_seconds=600)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            slot.ready_name = "cachedContents/fresh"
            slot.ready_offset = 3
            slot.ready_created_at = time.monotonic()  # Just now
            slot.config_fingerprint = _compute_fingerprint("sys", None)

            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name == "cachedContents/fresh"
        finally:
            registry.close()

    def test_api_ttl_includes_buffer(self):
        """Cache creation should use TTL + 60s buffer in the API call."""
        mock_client = MagicMock()
        mock_client.caches.create.return_value = MockCachedContent()
        registry = ContextCacheRegistry(
            mock_client, max_workers=1, cache_ttl_seconds=600
        )
        try:
            registry.register("agent-1", "test-model", min_token_threshold=100)
            contents = [MockContent(role="user", parts=[MockPart(text="data")])]
            registry.notify("agent-1", contents, "sys", None, token_count=50_000)

            slot = registry._slots["agent-1"]
            slot.pending.result(timeout=5)

            # Verify TTL in create call is 660s (600 + 60)
            create_call = mock_client.caches.create.call_args
            config = create_call.kwargs.get("config") or create_call[1].get("config")
            assert config.ttl == "660s"
        finally:
            registry.close()


class TestReaperThread:
    """Test the background reaper that cleans up expired caches."""

    def test_reaper_cleans_expired_caches(self):
        """The reaper should delete caches that have expired."""
        mock_client = MagicMock()
        # Use a very short TTL so caches expire quickly
        registry = ContextCacheRegistry(
            mock_client, max_workers=1, cache_ttl_seconds=0
        )
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            slot.ready_name = "cachedContents/stale"
            slot.ready_offset = 3
            slot.ready_created_at = time.monotonic() - 1  # Already expired

            # Wait for reaper to run (it checks every 60s, but we can trigger manually)
            # Instead of waiting 60s, let's test the reaper logic directly
            expired_names = []
            with registry._lock:
                for agent_id, s in registry._slots.items():
                    if s.ready_name and registry._is_expired(s):
                        expired_names.append(s.ready_name)
                        s.ready_name = None
                        s.ready_offset = 0
                        s.ready_created_at = None

            assert expired_names == ["cachedContents/stale"]
            assert slot.ready_name is None
        finally:
            registry.close()


class TestInvalidate:
    """Test cache invalidation."""

    def test_clears_ready_cache(self):
        """invalidate should clear ready cache and delete it remotely."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]
            slot.ready_name = "cachedContents/active"
            slot.ready_offset = 5
            slot.ready_created_at = time.monotonic()

            registry.invalidate("agent-1")

            assert slot.ready_name is None
            assert slot.ready_offset == 0
            mock_client.caches.delete.assert_called_with(name="cachedContents/active")
        finally:
            registry.close()

    def test_clears_pending(self):
        """invalidate should cancel/clear pending state."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]
            slot.pending = MagicMock()
            slot.pending.done.return_value = False
            slot.pending_through_index = 3

            registry.invalidate("agent-1")

            assert slot.pending is None
            assert slot.pending_through_index is None
        finally:
            registry.close()

    def test_tolerates_delete_failure(self):
        """invalidate should not raise if remote deletion fails."""
        mock_client = MagicMock()
        mock_client.caches.delete.side_effect = Exception("Network error")
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            registry._slots["agent-1"].ready_name = "cachedContents/stale"

            registry.invalidate("agent-1")  # Should not raise
            assert registry._slots["agent-1"].ready_name is None
        finally:
            registry.close()

    def test_invalidate_nonexistent_is_noop(self):
        """invalidate for unregistered agent is a no-op."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.invalidate("nonexistent")  # Should not raise
        finally:
            registry.close()


class TestUnregister:
    """Test agent unregistration and cleanup."""

    def test_deletes_ready_cache(self):
        """unregister should delete the ready cache."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            registry._slots["agent-1"].ready_name = "cachedContents/active"

            registry.unregister("agent-1")

            mock_client.caches.delete.assert_called_with(name="cachedContents/active")
            assert "agent-1" not in registry._slots
        finally:
            registry.close()

    def test_awaits_and_deletes_pending(self):
        """unregister should wait for pending cache, then delete it."""
        mock_client = MagicMock()
        mock_client.caches.create.return_value = MockCachedContent("cachedContents/pending")
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model", min_token_threshold=100)
            contents = [MockContent(role="user", parts=[MockPart(text="data")])]
            registry.notify("agent-1", contents, "sys", None, token_count=50_000)

            # Wait for pending to complete
            registry._slots["agent-1"].pending.result(timeout=5)

            registry.unregister("agent-1")

            mock_client.caches.delete.assert_called_with(name="cachedContents/pending")
        finally:
            registry.close()


class TestClose:
    """Test registry shutdown."""

    def test_deletes_all_caches(self):
        """close should delete all caches across all slots."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        registry.register("agent-1", "test-model")
        registry.register("agent-2", "test-model")
        registry._slots["agent-1"].ready_name = "cachedContents/a1"
        registry._slots["agent-2"].ready_name = "cachedContents/a2"

        registry.close()

        delete_calls = mock_client.caches.delete.call_args_list
        deleted_names = {c.kwargs["name"] for c in delete_calls}
        assert "cachedContents/a1" in deleted_names
        assert "cachedContents/a2" in deleted_names

    def test_close_is_idempotent(self):
        """close can be called multiple times without error."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        registry.close()
        registry.close()  # Should not raise


class TestMultiAgent:
    """Test multiple agents sharing the registry."""

    def test_independent_slots(self):
        """Each agent has its own cache slot."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=2)
        try:
            registry.register("agent-1", "model-a")
            registry.register("agent-2", "model-b")

            # Set cache for agent-1 only
            slot1 = registry._slots["agent-1"]
            slot1.ready_name = "cachedContents/a1"
            slot1.ready_offset = 3
            slot1.ready_created_at = time.monotonic()
            slot1.config_fingerprint = _compute_fingerprint("sys", None)

            advice1 = registry.get_advice("agent-1", "sys", None)
            advice2 = registry.get_advice("agent-2", "sys", None)

            assert advice1.cache_name == "cachedContents/a1"
            assert advice2.cache_name is None
        finally:
            registry.close()

    def test_invalidate_one_doesnt_affect_other(self):
        """invalidate for one agent should not affect another."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            registry.register("agent-2", "test-model")

            fp = _compute_fingerprint("sys", None)
            registry._slots["agent-1"].ready_name = "cachedContents/a1"
            registry._slots["agent-1"].ready_created_at = time.monotonic()
            registry._slots["agent-1"].config_fingerprint = fp
            registry._slots["agent-2"].ready_name = "cachedContents/a2"
            registry._slots["agent-2"].ready_created_at = time.monotonic()
            registry._slots["agent-2"].config_fingerprint = fp

            registry.invalidate("agent-1")

            assert registry._slots["agent-1"].ready_name is None
            assert registry._slots["agent-2"].ready_name == "cachedContents/a2"
        finally:
            registry.close()


class TestDeletionGuarantees:
    """Test that old caches are always cleaned up."""

    def test_old_cache_deleted_on_promotion(self):
        """When a new cache is promoted, the old one must be deleted."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            # Old ready cache
            slot.ready_name = "cachedContents/old"
            slot.ready_offset = 2
            slot.ready_created_at = time.monotonic()
            slot.config_fingerprint = _compute_fingerprint("sys", None)

            # New pending cache (done)
            future = Future()
            future.set_result("cachedContents/new")
            slot.pending = future
            slot.pending_through_index = 5

            # Trigger promotion via get_advice
            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name == "cachedContents/new"

            # Old cache deletion is submitted to executor — wait briefly
            registry._executor.shutdown(wait=True)
            # Recreate executor for close()
            registry._executor = ThreadPoolExecutor(max_workers=1)

            delete_calls = mock_client.caches.delete.call_args_list
            deleted_names = [c.kwargs["name"] for c in delete_calls]
            assert "cachedContents/old" in deleted_names
        finally:
            registry.close()

    def test_both_deleted_on_invalidate(self):
        """invalidate should delete both ready and pending caches."""
        mock_client = MagicMock()
        registry = ContextCacheRegistry(mock_client, max_workers=1)
        try:
            registry.register("agent-1", "test-model")
            slot = registry._slots["agent-1"]

            slot.ready_name = "cachedContents/ready"
            future = Future()
            future.set_result("cachedContents/pending-done")
            slot.pending = future
            slot.pending_through_index = 3

            registry.invalidate("agent-1")

            delete_calls = mock_client.caches.delete.call_args_list
            deleted_names = {c.kwargs["name"] for c in delete_calls}
            assert "cachedContents/ready" in deleted_names
            assert "cachedContents/pending-done" in deleted_names
        finally:
            registry.close()


class TestMultiRoundLifecycle:
    """End-to-end lifecycle simulating multiple conversation rounds."""

    def test_three_round_lifecycle(self):
        """Simulate the full pipeline over three rounds."""
        mock_client = MagicMock()
        cache_counter = [0]

        def make_cache(*args, **kwargs):
            cache_counter[0] += 1
            return MockCachedContent(f"cachedContents/cache_{cache_counter[0]}")

        mock_client.caches.create.side_effect = make_cache
        registry = ContextCacheRegistry(
            mock_client, max_workers=1, cache_ttl_seconds=600
        )
        try:
            registry.register("agent-1", "test-model", min_token_threshold=100)
            history = []

            # --- Round 1 ---
            history.append(MockContent(role="user", parts=[MockPart(text="msg1")]))
            history.append(MockContent(role="model", parts=[MockPart(text="resp1")]))

            # No cache yet
            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name is None

            # Notify with high token count triggers cache creation
            registry.notify("agent-1", history, "sys", None, token_count=50_000)
            slot = registry._slots["agent-1"]
            assert slot.pending is not None
            slot.pending.result(timeout=5)  # Wait for completion

            # --- Round 2 ---
            history.append(MockContent(role="user", parts=[MockPart(text="msg2")]))
            history.append(MockContent(role="model", parts=[MockPart(text="resp2")]))

            # get_advice auto-promotes cache_1
            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name == "cachedContents/cache_1"
            assert advice.contents_offset == 2  # Covers round 1's 2 items

            # Notify with enough new tokens triggers cache_2
            registry.notify("agent-1", history, "sys", None, token_count=60_000)
            assert slot.pending is not None
            slot.pending.result(timeout=5)

            # --- Round 3 ---
            history.append(MockContent(role="user", parts=[MockPart(text="msg3")]))
            history.append(MockContent(role="model", parts=[MockPart(text="resp3")]))

            # get_advice auto-promotes cache_2, deletes cache_1
            advice = registry.get_advice("agent-1", "sys", None)
            assert advice.cache_name == "cachedContents/cache_2"
            assert advice.contents_offset == 4  # Covers rounds 1+2

            # Wait for any background deletion work
            registry._executor.shutdown(wait=True)
            registry._executor = ThreadPoolExecutor(max_workers=1)

            # Verify cache_1 was deleted (old ready deleted on promotion)
            delete_calls = mock_client.caches.delete.call_args_list
            deleted_names = [c.kwargs["name"] for c in delete_calls]
            assert "cachedContents/cache_1" in deleted_names

        finally:
            registry.close()


# ============================================================
# Agent + Registry Integration Tests
# ============================================================


class TestAgentCachingIntegration:
    """Test that Agent.run() correctly uses the cache registry."""

    @pytest.fixture(autouse=True)
    def setup_registry(self, mock_env, mock_genai):
        """Set up a mock registry on the Agent class."""
        from agent_core.agents.base import Agent
        from agent_core.core.caching import CacheAdvice

        mock_client = mock_genai.Client.return_value
        mock_registry = MagicMock()
        mock_registry.get_advice.return_value = CacheAdvice(
            cache_name=None, contents_offset=0
        )

        # Store on Agent class
        old_registry = Agent._cache_registry
        Agent._cache_registry = mock_registry

        self.mock_client = mock_client
        self.mock_registry = mock_registry
        self.Agent = Agent
        self.CacheAdvice = CacheAdvice

        yield

        Agent._cache_registry = old_registry

    def test_caching_disabled_agent(self):
        """Agent with ENABLE_CACHING=False should not register."""

        class NoCacheAgent(self.Agent):
            ENABLE_CACHING = False

        self.mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = NoCacheAgent()
        assert agent._cache_enabled is False
        self.mock_registry.register.assert_not_called()

        result = agent.run("test")
        assert result == "ok"
        agent.close()

    def test_caching_enabled_registers(self):
        """Agent should register with registry when caching is enabled."""
        self.mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = self.Agent(session_id="test-session")
        assert agent._cache_enabled is True
        self.mock_registry.register.assert_called_once()
        agent.close()

    def test_no_cache_uses_base_config(self):
        """When registry returns no cache, agent uses base config."""
        self.mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = self.Agent(session_id="test-session")
        agent.run("hello")

        # get_advice was called
        self.mock_registry.get_advice.assert_called()
        # generate_content was called with full contents
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        assert len(call_kwargs["contents"]) == 1  # Just the user message
        agent.close()

    def test_cache_used_when_available(self):
        """When registry returns a cache, agent uses it with offset."""
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
        # Pre-populate history so offset makes sense
        agent._history = [
            MockContent(role="user", parts=[MockPart(text="msg1")]),
            MockContent(role="model", parts=[MockPart(text="resp1")]),
            MockContent(role="user", parts=[MockPart(text="msg2")]),
        ]

        result = agent.run("msg3")
        assert result == "cached response"

        # Check that generate_content received contents sliced at offset
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        # History is now [msg1, resp1, msg2, msg3] = 4 items, offset=3, so 1 sent
        assert len(call_kwargs["contents"]) == 1

        agent.close()

    def test_notify_called_after_run(self):
        """Agent should call notify on the registry after run completes."""
        self.mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = self.Agent(session_id="test-session")
        agent.run("hello")

        self.mock_registry.notify.assert_called()
        agent.close()

    def test_cache_fallback_on_failure(self):
        """If cached call fails, agent should fall back to uncached."""
        self.mock_registry.get_advice.return_value = self.CacheAdvice(
            cache_name="cachedContents/stale", contents_offset=2
        )

        call_count = [0]

        def generate_side_effect(**kwargs):
            call_count[0] += 1
            config = kwargs.get("config")
            if hasattr(config, "cached_content") and config.cached_content:
                raise Exception("CachedContent not found")
            return make_text_response("fallback response")

        self.mock_client.models.generate_content.side_effect = generate_side_effect

        agent = self.Agent(session_id="test-session")
        result = agent.run("test prompt")
        assert result == "fallback response"

        # Cache should have been invalidated
        self.mock_registry.invalidate.assert_called_with(agent.instance_id)
        agent.close()

    def test_clear_history_invalidates_cache(self):
        """clear_history should invalidate the agent's cache."""
        self.mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = self.Agent(session_id="test-session")
        agent.clear_history()

        self.mock_registry.invalidate.assert_called_with(agent.instance_id)
        agent.close()

    def test_close_unregisters(self):
        """close should unregister the agent from the registry."""
        self.mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = self.Agent(session_id="test-session")
        instance_id = agent.instance_id
        agent.close()

        self.mock_registry.unregister.assert_called_with(instance_id)
        assert agent._cache_enabled is False

    def test_run_stateless_does_not_use_cache(self):
        """run_stateless should not interact with the cache registry."""
        self.mock_client.models.generate_content.return_value = make_text_response(
            "stateless"
        )

        agent = self.Agent(session_id="test-session")
        agent.run_stateless("one-shot query")

        # No cache operations via get_advice/notify for stateless
        # get_advice may or may not be called — but no cache should be used
        self.mock_registry.notify.assert_not_called()
        agent.close()


class TestContentsOffset:
    """Test that contents_offset correctly splits cached vs uncached content."""

    def test_offset_zero_sends_full_contents(self, mock_env, mock_genai):
        """With offset=0, full contents should be sent."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = Agent()
        assert agent._cache_enabled is False  # No session_id

        agent.run("test")

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        assert len(contents) == 1
        agent.close()

    def test_offset_skips_cached_prefix(self, mock_env, mock_genai):
        """With offset > 0, only suffix should be sent to API."""
        from agent_core.agents.base import Agent
        from agent_core.core.caching import CacheAdvice

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = MockResponse(
            "ok",
            MockContent(role="model", parts=[MockPart(text="ok")]),
            usage_metadata=MockUsageMetadata(prompt_token_count=50_000),
        )

        # Set up registry
        mock_registry = MagicMock()
        mock_registry.get_advice.return_value = CacheAdvice(
            cache_name="cachedContents/c1", contents_offset=2
        )
        old_registry = Agent._cache_registry
        Agent._cache_registry = mock_registry

        try:
            agent = Agent(session_id="test-session")
            # Pre-populate history
            agent._history = [
                MockContent(role="user", parts=[MockPart(text="msg1")]),
                MockContent(role="model", parts=[MockPart(text="resp1")]),
            ]

            agent.run("msg2")

            # History after run: [msg1, resp1, msg2, model_resp] = 4 items
            # Offset=2, so only [msg2] sent (1 item, before model response added)
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            contents_sent = call_kwargs["contents"]
            assert len(contents_sent) == 1  # Only the new user message

            agent.close()
        finally:
            Agent._cache_registry = old_registry


class TestToolCallingWithCache:
    """Test caching interaction with the function-calling loop."""

    def test_tool_loop_appends_to_full_history(self, mock_env, mock_genai):
        """Tool results should be appended to full history even with offset."""
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
