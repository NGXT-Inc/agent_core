"""Tests for the CachePipeline and its integration with Agent.run().

Tests are organized into:
1. CachePipeline unit tests (isolated, no Agent)
2. Agent + caching integration tests (full pipeline through run())
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock, patch, call

import pytest

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
# CachePipeline Unit Tests
# ============================================================


class TestCachePipelineInit:
    """Test CachePipeline initialization and properties."""

    def test_initial_state(self):
        """Pipeline starts with no ready cache and no pending."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        pipeline = CachePipeline(mock_client, "gemini-3-pro-preview")

        assert pipeline.has_ready_cache is False
        assert pipeline.ready_cache_name is None
        assert pipeline.cached_through_index == 0

    def test_custom_threshold(self):
        """Minimum token threshold is configurable."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        pipeline = CachePipeline(mock_client, "gemini-3-pro-preview", min_token_threshold=8192)
        assert pipeline._min_token_threshold == 8192


class TestShouldCache:
    """Test the token threshold gating logic."""

    def test_below_threshold_returns_false(self):
        """Contents below token threshold should not be cached."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(10_000)
        pipeline = CachePipeline(mock_client, "test-model", min_token_threshold=32_768)

        contents = [MockContent(role="user", parts=[MockPart(text="hello")])]
        assert pipeline.should_cache(contents) is False

    def test_above_threshold_returns_true(self):
        """Contents above token threshold should be cached."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(50_000)
        pipeline = CachePipeline(mock_client, "test-model", min_token_threshold=32_768)

        contents = [MockContent(role="user", parts=[MockPart(text="large content")])]
        assert pipeline.should_cache(contents) is True

    def test_exact_threshold_returns_true(self):
        """Contents exactly at threshold should be cached."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(32_768)
        pipeline = CachePipeline(mock_client, "test-model", min_token_threshold=32_768)

        assert pipeline.should_cache([]) is True

    def test_known_token_count_skips_api_call(self):
        """When last_prompt_token_count is provided, count_tokens API is skipped."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        pipeline = CachePipeline(mock_client, "test-model", min_token_threshold=32_768)

        contents = [MockContent(role="user", parts=[MockPart(text="hello")])]

        # Above threshold via known count
        assert pipeline.should_cache(contents, last_prompt_token_count=50_000) is True
        mock_client.models.count_tokens.assert_not_called()

        # Below threshold via known count
        assert pipeline.should_cache(contents, last_prompt_token_count=1_000) is False
        mock_client.models.count_tokens.assert_not_called()

    def test_count_tokens_failure_returns_false(self):
        """If count_tokens API fails, should_cache returns False gracefully."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.models.count_tokens.side_effect = Exception("API Error")
        pipeline = CachePipeline(mock_client, "test-model")

        assert pipeline.should_cache([]) is False


class TestCreateCacheAsync:
    """Test background cache creation."""

    def test_submits_to_executor(self):
        """create_cache_async submits work to the provided executor."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.caches.create.return_value = MockCachedContent("cachedContents/abc")
        pipeline = CachePipeline(mock_client, "test-model")

        contents = [MockContent(role="user", parts=[MockPart(text="test")])]

        with ThreadPoolExecutor(max_workers=1) as executor:
            pipeline.create_cache_async(
                contents=contents,
                system_instruction="You are helpful.",
                tools=None,
                executor=executor,
            )

            assert pipeline._pending is not None
            assert pipeline._pending_through_index == 1

            # Wait for it to complete
            result = pipeline._pending.result(timeout=5)
            assert result == "cachedContents/abc"

    def test_skips_if_already_pending(self):
        """Should not fire a second cache creation if one is already pending."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.caches.create.return_value = MockCachedContent()
        pipeline = CachePipeline(mock_client, "test-model")

        with ThreadPoolExecutor(max_workers=1) as executor:
            pipeline.create_cache_async([], None, None, executor)
            first_pending = pipeline._pending

            # Second call should be skipped
            pipeline.create_cache_async([], None, None, executor)
            assert pipeline._pending is first_pending

    def test_records_through_index(self):
        """Should record the number of content items being cached."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.caches.create.return_value = MockCachedContent()
        pipeline = CachePipeline(mock_client, "test-model")

        contents = [
            MockContent(role="user", parts=[MockPart(text="msg1")]),
            MockContent(role="model", parts=[MockPart(text="resp1")]),
            MockContent(role="user", parts=[MockPart(text="msg2")]),
        ]

        with ThreadPoolExecutor(max_workers=1) as executor:
            pipeline.create_cache_async(contents, None, None, executor)
            assert pipeline._pending_through_index == 3


class TestPromotePending:
    """Test the promote_pending transition."""

    def test_promotes_on_success(self):
        """Successful pending cache becomes the ready cache."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.caches.create.return_value = MockCachedContent("cachedContents/new")
        pipeline = CachePipeline(mock_client, "test-model")

        with ThreadPoolExecutor(max_workers=1) as executor:
            pipeline.create_cache_async(
                [MockContent(role="user", parts=[MockPart(text="x")])],
                None, None, executor,
            )
            pipeline._pending.result(timeout=5)  # ensure complete

            pipeline.promote_pending()

        assert pipeline.has_ready_cache is True
        assert pipeline.ready_cache_name == "cachedContents/new"
        assert pipeline.cached_through_index == 1
        assert pipeline._pending is None

    def test_deletes_old_ready_on_promote(self):
        """When promoting, the old ready cache should be deleted."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        pipeline = CachePipeline(mock_client, "test-model")

        # Manually set an existing ready cache
        pipeline._ready_name = "cachedContents/old"
        pipeline._cached_through_index = 2

        # Create and complete a pending cache
        mock_client.caches.create.return_value = MockCachedContent("cachedContents/new")
        with ThreadPoolExecutor(max_workers=1) as executor:
            pipeline.create_cache_async(
                [MockContent(role="user", parts=[MockPart(text="x")])] * 5,
                None, None, executor,
            )
            pipeline._pending.result(timeout=5)

            pipeline.promote_pending()

        # Old cache should have been deleted
        mock_client.caches.delete.assert_called_once_with(name="cachedContents/old")
        # New cache is ready
        assert pipeline.ready_cache_name == "cachedContents/new"
        assert pipeline.cached_through_index == 5

    def test_handles_creation_failure(self):
        """If cache creation fails, ready state is unchanged."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.caches.create.side_effect = Exception("API quota exceeded")
        pipeline = CachePipeline(mock_client, "test-model")

        # Set existing ready cache
        pipeline._ready_name = "cachedContents/existing"
        pipeline._cached_through_index = 3

        with ThreadPoolExecutor(max_workers=1) as executor:
            pipeline.create_cache_async([], None, None, executor)
            # Wait for the future to complete (with error)
            try:
                pipeline._pending.result(timeout=5)
            except Exception:
                pass

            pipeline.promote_pending()

        # Ready cache should be unchanged (old one still valid)
        assert pipeline.ready_cache_name == "cachedContents/existing"
        assert pipeline.cached_through_index == 3
        assert pipeline._pending is None

    def test_noop_when_no_pending(self):
        """promote_pending should be a no-op when nothing is pending."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        pipeline = CachePipeline(mock_client, "test-model")

        pipeline.promote_pending()  # Should not raise
        assert pipeline.has_ready_cache is False


class TestInvalidate:
    """Test cache invalidation."""

    def test_clears_ready_cache(self):
        """invalidate() should clear ready cache and delete it remotely."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        pipeline = CachePipeline(mock_client, "test-model")
        pipeline._ready_name = "cachedContents/active"
        pipeline._cached_through_index = 5

        pipeline.invalidate()

        assert pipeline.has_ready_cache is False
        assert pipeline.cached_through_index == 0
        mock_client.caches.delete.assert_called_once_with(name="cachedContents/active")

    def test_clears_pending(self):
        """invalidate() should clear pending state."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        pipeline = CachePipeline(mock_client, "test-model")
        pipeline._pending = MagicMock()
        pipeline._pending_through_index = 3

        pipeline.invalidate()

        assert pipeline._pending is None
        assert pipeline._pending_through_index is None

    def test_tolerates_delete_failure(self):
        """invalidate() should not raise if remote deletion fails."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.caches.delete.side_effect = Exception("Network error")
        pipeline = CachePipeline(mock_client, "test-model")
        pipeline._ready_name = "cachedContents/stale"

        pipeline.invalidate()  # Should not raise
        assert pipeline.has_ready_cache is False


class TestCleanup:
    """Test resource cleanup on agent teardown."""

    def test_deletes_ready_cache(self):
        """cleanup() should delete the ready cache."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        pipeline = CachePipeline(mock_client, "test-model")
        pipeline._ready_name = "cachedContents/active"

        pipeline.cleanup()

        mock_client.caches.delete.assert_called_with(name="cachedContents/active")
        assert pipeline._ready_name is None

    def test_waits_for_pending_and_deletes(self):
        """cleanup() should wait for pending cache, then delete it."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        mock_client.caches.create.return_value = MockCachedContent("cachedContents/pending")
        pipeline = CachePipeline(mock_client, "test-model")

        with ThreadPoolExecutor(max_workers=1) as executor:
            pipeline.create_cache_async([], None, None, executor)
            pipeline._pending.result(timeout=5)

            pipeline.cleanup()

        mock_client.caches.delete.assert_called_with(name="cachedContents/pending")


class TestPipelineLifecycle:
    """End-to-end pipeline lifecycle tests simulating multiple rounds."""

    def test_three_round_pipeline(self):
        """Simulate the full i-2 pipeline over three rounds."""
        from agent_core.core.caching import CachePipeline

        mock_client = MagicMock()
        cache_counter = [0]

        def make_cache(*args, **kwargs):
            cache_counter[0] += 1
            return MockCachedContent(f"cachedContents/cache_{cache_counter[0]}")

        mock_client.caches.create.side_effect = make_cache
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(50_000)

        pipeline = CachePipeline(mock_client, "test-model", min_token_threshold=32_768)
        history = []

        with ThreadPoolExecutor(max_workers=1) as executor:
            # --- Round 1 ---
            history.append(MockContent(role="user", parts=[MockPart(text="msg1")]))
            history.append(MockContent(role="model", parts=[MockPart(text="resp1")]))

            # No ready cache yet
            assert pipeline.has_ready_cache is False

            # Post-round 1: promote (no-op) + fire cache
            pipeline.promote_pending()
            assert pipeline.has_ready_cache is False

            pipeline.create_cache_async(list(history), "sys", None, executor)
            pipeline._pending.result(timeout=5)  # let it finish

            # --- Round 2 ---
            history.append(MockContent(role="user", parts=[MockPart(text="msg2")]))
            history.append(MockContent(role="model", parts=[MockPart(text="resp2")]))

            # Still no ready cache (pending from round 1 not promoted yet)
            assert pipeline.has_ready_cache is False

            # Post-round 2: promote cache_1 → ready, fire cache_2
            pipeline.promote_pending()
            assert pipeline.has_ready_cache is True
            assert pipeline.ready_cache_name == "cachedContents/cache_1"
            assert pipeline.cached_through_index == 2  # covers 2 items from round 1

            pipeline.create_cache_async(list(history), "sys", None, executor)
            pipeline._pending.result(timeout=5)

            # --- Round 3 ---
            history.append(MockContent(role="user", parts=[MockPart(text="msg3")]))
            history.append(MockContent(role="model", parts=[MockPart(text="resp3")]))

            # Ready cache from round 1 (i-2)
            assert pipeline.ready_cache_name == "cachedContents/cache_1"
            assert pipeline.cached_through_index == 2

            # Post-round 3: promote cache_2 → ready (deletes cache_1), fire cache_3
            pipeline.promote_pending()
            assert pipeline.ready_cache_name == "cachedContents/cache_2"
            assert pipeline.cached_through_index == 4  # covers 4 items from rounds 1+2

            # cache_1 should have been deleted
            mock_client.caches.delete.assert_called_with(name="cachedContents/cache_1")

            pipeline.create_cache_async(list(history), "sys", None, executor)
            pipeline._pending.result(timeout=5)

        # Cleanup
        pipeline.cleanup()


# ============================================================
# Agent + Caching Integration Tests
# ============================================================


class TestAgentCachingIntegration:
    """Test that Agent.run() correctly uses the caching pipeline."""

    @pytest.fixture
    def agent_with_caching(self, mock_env, mock_genai):
        """Create an Agent with caching enabled and mocked internals."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        # First run: text response
        mock_client.models.generate_content.return_value = make_text_response("Hello!")
        # Below threshold initially
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(1000)

        agent = Agent(session_id="test-session")
        yield agent, mock_client
        agent.close()

    def test_caching_disabled_agent(self, mock_env, mock_genai):
        """Agent with ENABLE_CACHING=False should not create a pipeline."""
        from agent_core.agents.base import Agent

        class NoCacheAgent(Agent):
            ENABLE_CACHING = False

        mock_genai.Client.return_value.models.generate_content.return_value = (
            make_text_response("ok")
        )

        agent = NoCacheAgent()
        assert agent._cache_pipeline is None
        assert agent._cache_executor is None

        result = agent.run("test")
        assert result == "ok"
        agent.close()

    def test_caching_enabled_by_default(self, mock_env, mock_genai):
        """Agent should have caching enabled by default when session_id is set."""
        from agent_core.agents.base import Agent

        mock_genai.Client.return_value.models.generate_content.return_value = (
            make_text_response("ok")
        )

        agent = Agent(session_id="test-session")
        assert agent._cache_pipeline is not None
        assert agent._cache_executor is not None
        agent.close()

    def test_no_cache_below_threshold(self, agent_with_caching):
        """When content is below threshold, no cache should be created."""
        agent, mock_client = agent_with_caching
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(1000)

        agent.run("hello")

        # No cache should be created (below threshold)
        mock_client.caches.create.assert_not_called()

    def test_cache_created_above_threshold(self, agent_with_caching):
        """When content exceeds threshold after run, cache should be fired."""
        agent, mock_client = agent_with_caching
        # Response reports prompt tokens above threshold — should_cache uses this
        mock_client.models.generate_content.return_value = MockResponse(
            "Hello!",
            MockContent(role="model", parts=[MockPart(text="Hello!")]),
            usage_metadata=MockUsageMetadata(prompt_token_count=50_000),
        )

        agent.run("large prompt with lots of context")

        # Cache should have been created
        mock_client.caches.create.assert_called_once()

    def test_cache_used_on_subsequent_run(self, mock_env, mock_genai):
        """After cache is created and promoted, next run should use it."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        # Response reports prompt tokens above threshold so should_cache triggers
        mock_client.models.generate_content.return_value = MockResponse(
            "response",
            MockContent(role="model", parts=[MockPart(text="response")]),
            usage_metadata=MockUsageMetadata(prompt_token_count=50_000),
        )

        cache_name = "cachedContents/test-123"
        mock_client.caches.create.return_value = MockCachedContent(cache_name)

        agent = Agent(session_id="test-session")

        # Round 1: no cache yet, fires cache creation
        agent.run("first message")

        # Manually wait for the pending cache to complete
        pipeline = agent._cache_pipeline
        if pipeline._pending:
            pipeline._pending.result(timeout=5)

        # Round 2: promote pending → ready, but no ready yet for THIS run
        agent.run("second message")

        # Now pipeline should have ready cache
        assert pipeline.has_ready_cache is True

        # Round 3: should use the ready cache
        agent.run("third message")

        # Check that generate_content was called with cached_content config
        calls = mock_client.models.generate_content.call_args_list
        # The third call should have cached_content in its config
        third_call_config = calls[2].kwargs.get("config")
        assert third_call_config is not None

        agent.close()

    def test_clear_history_invalidates_cache(self, agent_with_caching):
        """clear_history() should invalidate the cache pipeline."""
        agent, mock_client = agent_with_caching

        # Manually set a ready cache
        agent._cache_pipeline._ready_name = "cachedContents/active"
        agent._cache_pipeline._cached_through_index = 5

        agent.clear_history()

        assert agent._cache_pipeline.has_ready_cache is False
        assert agent._cache_pipeline.cached_through_index == 0
        mock_client.caches.delete.assert_called_with(name="cachedContents/active")

    def test_invalidate_cache_method(self, agent_with_caching):
        """_invalidate_cache() should clear the pipeline."""
        agent, mock_client = agent_with_caching

        agent._cache_pipeline._ready_name = "cachedContents/active"
        agent._invalidate_cache()

        assert agent._cache_pipeline.has_ready_cache is False

    def test_close_cleans_up_resources(self, mock_env, mock_genai):
        """close() should clean up cache and executor."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = Agent(session_id="test-session")
        pipeline = agent._cache_pipeline
        executor = agent._cache_executor

        agent.close()

        assert agent._cache_executor is None

    def test_cache_fallback_on_stale_cache(self, mock_env, mock_genai):
        """If cached generate_content fails, should fall back to uncached."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(50_000)
        mock_client.caches.create.return_value = MockCachedContent("cachedContents/stale")

        call_count = [0]

        def generate_side_effect(**kwargs):
            call_count[0] += 1
            config = kwargs.get("config")
            # If using cached_content, simulate cache expiry error
            if hasattr(config, "cached_content") and config.cached_content:
                raise Exception("CachedContent not found")
            return make_text_response("fallback response")

        mock_client.models.generate_content.side_effect = generate_side_effect

        agent = Agent(session_id="test-session")

        # Manually set up a ready cache to trigger cached path
        agent._cache_pipeline._ready_name = "cachedContents/stale"
        agent._cache_pipeline._cached_through_index = 0

        # Should fall back to uncached and succeed
        result = agent.run("test prompt")
        assert result == "fallback response"

        # Cache should have been invalidated
        assert agent._cache_pipeline.has_ready_cache is False

        agent.close()

    def test_run_stateless_does_not_use_cache(self, mock_env, mock_genai):
        """run_stateless() should not interact with the cache pipeline."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("stateless")
        mock_client.models.count_tokens.return_value = MockTokenCountResponse(50_000)

        agent = Agent(session_id="test-session")

        agent.run_stateless("one-shot query")

        # No cache operations should have happened
        mock_client.caches.create.assert_not_called()

        agent.close()


class TestContentsOffset:
    """Test that contents_offset correctly splits cached vs uncached content."""

    def test_offset_zero_sends_full_contents(self, mock_env, mock_genai):
        """With offset=0, full contents should be sent."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = Agent()
        agent.ENABLE_CACHING = False
        agent._cache_pipeline = None

        agent.run("test")

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        # Should be the full history (1 user message)
        assert len(contents) == 1

        agent.close()

    def test_offset_skips_cached_prefix(self, mock_env, mock_genai):
        """With offset > 0, only suffix should be sent to API."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        # Response reports prompt tokens above threshold so should_cache triggers
        mock_client.models.generate_content.return_value = MockResponse(
            "ok",
            MockContent(role="model", parts=[MockPart(text="ok")]),
            usage_metadata=MockUsageMetadata(prompt_token_count=50_000),
        )
        mock_client.caches.create.return_value = MockCachedContent("cachedContents/c1")

        agent = Agent(session_id="test-session")

        # Round 1
        agent.run("first")

        # Wait for cache creation
        pipeline = agent._cache_pipeline
        if pipeline._pending:
            pipeline._pending.result(timeout=5)

        # Round 2 - promotes cache, fires new
        agent.run("second")

        # Wait for cache creation
        if pipeline._pending:
            pipeline._pending.result(timeout=5)

        # Now cache is ready, covering first 2 history items (user+model from round 1)
        assert pipeline.has_ready_cache is True
        cached_through = pipeline.cached_through_index

        # Round 3 - should use cache
        agent.run("third")

        # The last generate_content call should have received only uncached items
        last_call = mock_client.models.generate_content.call_args
        contents_sent = last_call.kwargs["contents"]

        # Contents sent should be fewer than full history.
        # At call time, history had: u1,m1,u2,m2,u3 (5 items, before model reply).
        # cached_through=2 (covers u1,m1), so 5-2=3 items sent.
        full_history_len = len(agent._history)  # 7 after model reply added
        assert len(contents_sent) < full_history_len
        assert len(contents_sent) == (full_history_len - 1) - cached_through

        agent.close()


class TestToolCallingWithCache:
    """Test caching interaction with the function-calling loop."""

    def test_tool_loop_appends_to_full_history(self, mock_env, mock_genai):
        """Tool results should be appended to full history even with offset."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        # Response sequence: tool call → text response
        responses = [
            make_tool_call_response("my_tool", {"query": "test"}),
            make_text_response("Done!"),
        ]
        mock_client.models.generate_content.side_effect = responses

        agent = Agent()
        agent.ENABLE_CACHING = False
        agent._cache_pipeline = None

        def my_tool(query: str) -> str:
            """A test tool."""
            return f"result for {query}"

        agent.register_tool(my_tool)
        result = agent.run("use the tool")

        assert result == "Done!"
        # History should contain: user msg, model tool call, tool response, model text
        assert len(agent._history) == 4

        agent.close()
