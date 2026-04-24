"""Tests for the Agent base class (excluding caching, tested in test_caching.py)."""

import os
import time
from unittest.mock import MagicMock, patch, call

import pytest

from tests.conftest import (
    MockContent,
    MockPart,
    MockFunctionCall,
    MockResponse,
    MockTokenCountResponse,
    MockUsageMetadata,
    make_text_response,
    make_tool_call_response,
    make_multi_tool_call_response,
)


class TestAgentInit:
    """Test Agent initialization."""

    def test_requires_google_project_id(self):
        """Should raise if GOOGLE_PROJECT_ID is not set."""
        from agent_core.agents.base import Agent

        with patch.dict(os.environ, {"GOOGLE_PROJECT_ID": ""}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_PROJECT_ID"):
                Agent()

    def test_default_model(self, mock_env, mock_genai):
        """Should use DEFAULT_MODEL when no model_name is provided."""
        from agent_core.agents.base import Agent

        agent = Agent()
        assert agent.model_name == Agent.DEFAULT_MODEL
        agent.close()

    def test_custom_model(self, mock_env, mock_genai):
        """Should use provided model_name."""
        from agent_core.agents.base import Agent

        agent = Agent(model_name="gemini-custom")
        assert agent.model_name == "gemini-custom"
        agent.close()

    def test_instance_id_generated(self, mock_env, mock_genai):
        """Should generate a unique instance ID."""
        from agent_core.agents.base import Agent

        agent = Agent()
        assert agent.instance_id.startswith("base_")
        agent.close()

    def test_session_based_id_for_root_agents(self, mock_env, mock_genai):
        """Root agent types should get deterministic IDs from session_id."""
        from agent_core.agents.base import Agent

        class DesignerAgent(Agent):
            name = "designer"

        agent = DesignerAgent(session_id="abcdef1234567890")
        assert agent.instance_id == "designer_abcdef12"
        agent.close()

    def test_empty_history_on_init(self, mock_env, mock_genai):
        """History should be empty on fresh initialization."""
        from agent_core.agents.base import Agent

        agent = Agent()
        assert agent._history == []
        agent.close()

    def test_injected_client_used(self, mock_client):
        """Should use the injected client and not create a new one."""
        from agent_core.agents.base import Agent

        agent = Agent(client=mock_client)
        assert agent.client is mock_client
        agent.close()

    def test_injected_client_skips_project_id_check(self, mock_client):
        """Should not require GOOGLE_PROJECT_ID when client is injected."""
        from agent_core.agents.base import Agent

        with patch.dict(os.environ, {"GOOGLE_PROJECT_ID": ""}, clear=True):
            agent = Agent(client=mock_client)
            assert agent.client is mock_client
            agent.close()


class TestToolRegistration:
    """Test tool registration and execution."""

    def test_register_tool(self, mock_env, mock_genai):
        """Should register a callable tool."""
        from agent_core.agents.base import Agent

        agent = Agent()

        def my_tool(query: str) -> str:
            """Search for something."""
            return f"found: {query}"

        agent.register_tool(my_tool)
        assert "my_tool" in agent._tools
        assert my_tool in agent._tool_functions
        agent.close()

    def test_tool_execution_through_run(self, mock_env, mock_genai):
        """Tool should be called when model requests it."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        # Sequence: tool call → final text
        responses = [
            make_tool_call_response("search", {"query": "python"}),
            make_text_response("Found Python docs."),
        ]
        mock_client.models.generate_content.side_effect = responses

        agent = Agent()
        tool_called_with = {}

        def search(query: str) -> str:
            """Search tool."""
            tool_called_with["query"] = query
            return "Python documentation"

        agent.register_tool(search)
        result = agent.run("search for python")

        assert tool_called_with["query"] == "python"
        assert result == "Found Python docs."
        agent.close()

    def test_parallel_tool_execution(self, mock_env, mock_genai):
        """Multiple tool calls in one response should execute in parallel."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        responses = [
            make_multi_tool_call_response([
                ("tool_a", {"x": "1"}),
                ("tool_b", {"y": "2"}),
            ]),
            make_text_response("Both done."),
        ]
        mock_client.models.generate_content.side_effect = responses

        agent = Agent()
        calls = []

        def tool_a(x: str) -> str:
            """Tool A."""
            calls.append(("a", x))
            return "result_a"

        def tool_b(y: str) -> str:
            """Tool B."""
            calls.append(("b", y))
            return "result_b"

        agent.register_tool(tool_a)
        agent.register_tool(tool_b)
        result = agent.run("use both tools")

        assert ("a", "1") in calls
        assert ("b", "2") in calls
        assert result == "Both done."
        agent.close()

    def test_unknown_tool_returns_error(self, mock_env, mock_genai):
        """Calling an unregistered tool should return an error dict."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        responses = [
            make_tool_call_response("nonexistent_tool", {}),
            make_text_response("ok"),
        ]
        mock_client.models.generate_content.side_effect = responses

        agent = Agent()
        result = agent.run("test")
        assert result == "ok"
        agent.close()

    def test_tool_error_caught(self, mock_env, mock_genai):
        """Tool exceptions should be caught and returned as error results."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        responses = [
            make_tool_call_response("failing_tool", {}),
            make_text_response("Handled the error."),
        ]
        mock_client.models.generate_content.side_effect = responses

        agent = Agent()

        def failing_tool() -> str:
            """A tool that fails."""
            raise RuntimeError("Tool broke!")

        agent.register_tool(failing_tool)
        result = agent.run("use the tool")

        # Should not crash, model should get error and respond
        assert result == "Handled the error."
        agent.close()


class TestToolUnregistration:
    """Test tool unregistration."""

    def test_unregister_tool(self, mock_env, mock_genai):
        """Should remove tool from all internal registries."""
        from agent_core.agents.base import Agent

        agent = Agent()

        def my_tool(x: str) -> str:
            """A tool."""
            return x

        agent.register_tool(my_tool)
        assert "my_tool" in agent._tools

        agent.unregister_tool("my_tool")
        assert "my_tool" not in agent._tools
        assert my_tool not in agent._tool_functions
        agent.close()

    def test_unregister_unknown_tool_raises(self, mock_env, mock_genai):
        """Should raise KeyError for unknown tool name."""
        from agent_core.agents.base import Agent

        agent = Agent()
        with pytest.raises(KeyError, match="no_such_tool"):
            agent.unregister_tool("no_such_tool")
        agent.close()

    def test_unregister_then_register(self, mock_env, mock_genai):
        """Should be able to register a new tool after unregistering one."""
        from agent_core.agents.base import Agent

        agent = Agent()

        def tool_a(x: str) -> str:
            """Tool A."""
            return x

        def tool_b(y: str) -> str:
            """Tool B."""
            return y

        agent.register_tool(tool_a)
        agent.unregister_tool("tool_a")
        agent.register_tool(tool_b)

        assert "tool_a" not in agent._tools
        assert "tool_b" in agent._tools
        assert len(agent._tool_functions) == 1
        agent.close()

    def test_unregister_last_tool(self, mock_env, mock_genai):
        """Unregistering the only tool should leave empty declarations."""
        from agent_core.agents.base import Agent

        agent = Agent()

        def only_tool(x: str) -> str:
            """The only tool."""
            return x

        agent.register_tool(only_tool)
        agent.unregister_tool("only_tool")

        assert agent._tool_schemas is None
        assert agent._tool_functions == []
        assert agent._tools == {}
        agent.close()


class TestConversationHistory:
    """Test history management in run() and run_stateless()."""

    def test_run_accumulates_history(self, mock_env, mock_genai):
        """run() should accumulate user and model messages."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("reply")

        agent = Agent()
        agent.run("msg1")
        agent.run("msg2")

        assert len(agent._history) == 4  # user1, model1, user2, model2
        agent.close()

    def test_run_stateless_no_history(self, mock_env, mock_genai):
        """run_stateless() should not affect history."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("reply")

        agent = Agent()
        agent.run_stateless("one-shot")

        assert len(agent._history) == 0
        agent.close()

    def test_clear_history(self, mock_env, mock_genai):
        """clear_history() should empty the history."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("reply")

        agent = Agent()
        agent.run("hello")
        assert len(agent._history) > 0

        agent.clear_history()
        assert len(agent._history) == 0
        agent.close()

    def test_history_with_persistence(self, mock_env, mock_genai):
        """History should be saved to conversation store when provided."""
        from agent_core.agents.base import Agent
        from agent_core.core.persistence import InMemoryConversationStore

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("reply")

        store = InMemoryConversationStore()
        agent = Agent(session_id="test-session", conversation_store=store)
        agent.run("hello")

        saved = store.load("test-session", "base")
        assert len(saved) > 0
        agent.close()

    def test_run_stateless_does_not_save_history(self, mock_env, mock_genai):
        """run_stateless() should never call _save_history()."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        # Multi-iteration: tool call followed by final text response
        responses = [
            make_tool_call_response("my_tool", {"x": "1"}),
            make_text_response("done"),
        ]
        mock_client.models.generate_content.side_effect = responses

        store = MagicMock()
        agent = Agent(session_id="sess", conversation_store=store)

        def my_tool(x: str) -> str:
            """A tool."""
            return "ok"

        agent.register_tool(my_tool)
        agent.run_stateless("go")

        store.save.assert_not_called()
        agent.close()

    def test_run_saves_history_during_tool_loop(self, mock_env, mock_genai):
        """run() should call _save_history() during the tool loop."""
        from agent_core.agents.base import Agent
        from agent_core.core.persistence import InMemoryConversationStore

        mock_client = mock_genai.Client.return_value

        responses = [
            make_tool_call_response("my_tool", {"x": "1"}),
            make_text_response("done"),
        ]
        mock_client.models.generate_content.side_effect = responses

        store = InMemoryConversationStore()
        agent = Agent(session_id="sess", conversation_store=store)

        def my_tool(x: str) -> str:
            """A tool."""
            return "ok"

        agent.register_tool(my_tool)
        agent.run("go")

        saved = store.load("sess", "base")
        # user msg + model tool call + tool response + model final = 4
        assert len(saved) == 4
        agent.close()

    def test_run_streams_text_deltas(self, mock_env, mock_genai):
        """run() should use provider streaming when explicitly requested."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content_stream.return_value = iter([
            make_text_response("Hel"),
            make_text_response("lo"),
        ])

        deltas = []
        agent = Agent()
        result = agent.run("hello", streaming=True, on_text_delta=deltas.append)

        assert result == "Hello"
        assert deltas == ["Hel", "lo"]
        mock_client.models.generate_content.assert_not_called()
        assert len(agent._history) == 2
        agent.close()

    def test_run_streaming_true_without_callback_uses_stream_transport(self, mock_env, mock_genai):
        """streaming=True should not require a text callback to take effect."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content_stream.return_value = iter([
            make_text_response("streamed"),
        ])

        agent = Agent()
        result = agent.run("hello", streaming=True)

        assert result == "streamed"
        mock_client.models.generate_content.assert_not_called()
        mock_client.models.generate_content_stream.assert_called_once()
        agent.close()

    def test_agent_streaming_default_can_be_overridden_per_run(self, mock_env, mock_genai):
        """Constructor default controls streaming unless run() overrides it."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("plain")

        agent = Agent(streaming=True)
        result = agent.run("hello", streaming=False)

        assert result == "plain"
        mock_client.models.generate_content.assert_called_once()
        mock_client.models.generate_content_stream.assert_not_called()
        agent.close()

    def test_agent_streaming_default_applies_to_run(self, mock_env, mock_genai):
        """Agent-level streaming default should apply when run() omits streaming."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content_stream.return_value = iter([
            make_text_response("default stream"),
        ])

        agent = Agent(streaming=True)
        result = agent.run("hello")

        assert result == "default stream"
        mock_client.models.generate_content.assert_not_called()
        mock_client.models.generate_content_stream.assert_called_once()
        agent.close()

    def test_class_streaming_default_applies_to_run(self, mock_env, mock_genai):
        """DEFAULT_STREAMING should configure streaming for subclasses."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content_stream.return_value = iter([
            make_text_response("class stream"),
        ])

        class StreamingAgent(Agent):
            DEFAULT_STREAMING = True

        agent = StreamingAgent()
        result = agent.run("hello")

        assert result == "class stream"
        mock_client.models.generate_content.assert_not_called()
        mock_client.models.generate_content_stream.assert_called_once()
        agent.close()

    def test_agent_streaming_default_applies_to_run_stateless(self, mock_env, mock_genai):
        """Agent-level streaming default should apply to stateless calls too."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content_stream.return_value = iter([
            make_text_response("stateless stream"),
        ])

        agent = Agent(streaming=True)
        result = agent.run_stateless("hello")

        assert result == "stateless stream"
        mock_client.models.generate_content.assert_not_called()
        mock_client.models.generate_content_stream.assert_called_once()
        agent.close()


class TestLifecycleHooks:
    """Test that lifecycle hooks are called correctly."""

    def test_on_agent_start_called(self, mock_env, mock_genai):
        """on_agent_start should be called with the prompt."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        class HookedAgent(Agent):
            started_with = None
            def on_agent_start(self, prompt):
                HookedAgent.started_with = prompt

        agent = HookedAgent()
        agent.run("test prompt")

        assert HookedAgent.started_with == "test prompt"
        agent.close()

    def test_on_agent_end_called_on_success(self, mock_env, mock_genai):
        """on_agent_end should be called with success=True."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        end_args = {}

        class HookedAgent(Agent):
            def on_agent_end(self, result, success, error=None, cancelled=False):
                end_args["result"] = result
                end_args["success"] = success
                end_args["error"] = error
                end_args["cancelled"] = cancelled

        agent = HookedAgent()
        agent.run("test")

        assert end_args["success"] is True
        assert end_args["result"] == "ok"
        assert end_args["error"] is None
        assert end_args["cancelled"] is False
        agent.close()

    def test_on_agent_end_called_on_failure(self, mock_env, mock_genai):
        """on_agent_end should be called with success=False on error."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.side_effect = RuntimeError("API down")

        end_args = {}

        class HookedAgent(Agent):
            def on_agent_end(self, result, success, error=None):
                end_args["success"] = success
                end_args["error"] = error

        agent = HookedAgent()
        with pytest.raises(RuntimeError):
            agent.run("test")

        assert end_args["success"] is False
        assert "API down" in end_args["error"]
        agent.close()

    def test_on_agent_end_cancelled_flag(self, mock_env, mock_genai):
        """on_agent_end should receive cancelled=True when cancelled."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_tool_call_response(
            "my_tool", {}
        )

        end_args = {}

        class HookedAgent(Agent):
            def on_agent_end(self, result, success, error=None, cancelled=False):
                end_args["success"] = success
                end_args["cancelled"] = cancelled

        agent = HookedAgent()

        def my_tool() -> str:
            """Tool that cancels."""
            agent.cancel()
            return "ok"

        agent.register_tool(my_tool)
        agent.run("go")

        assert end_args["cancelled"] is True
        assert end_args["success"] is False
        agent.close()

    def test_on_tool_hooks_called(self, mock_env, mock_genai):
        """on_tool_start and on_tool_end should be called for each tool."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        responses = [
            make_tool_call_response("my_tool", {"arg": "val"}),
            make_text_response("done"),
        ]
        mock_client.models.generate_content.side_effect = responses

        hook_calls = []

        class HookedAgent(Agent):
            def on_tool_start(self, tool_name, args, tool_call_id):
                hook_calls.append(("start", tool_name))

            def on_tool_end(self, tool_name, args, tool_call_id, result, success, error=None):
                hook_calls.append(("end", tool_name, success))

        agent = HookedAgent()

        def my_tool(arg: str) -> str:
            """Test tool."""
            return "result"

        agent.register_tool(my_tool)
        agent.run("use the tool")

        assert ("start", "my_tool") in hook_calls
        assert ("end", "my_tool", True) in hook_calls
        agent.close()


class TestMaxIterations:
    """Test the iteration limit safety valve."""

    def test_max_iterations_reached(self, mock_env, mock_genai):
        """Should stop after MAX_ITERATIONS and return a message."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        # Always return tool calls (never text)
        mock_client.models.generate_content.return_value = make_tool_call_response(
            "infinite_tool", {}
        )

        class LimitedAgent(Agent):
            MAX_ITERATIONS = 3

        agent = LimitedAgent()

        def infinite_tool() -> str:
            """Never-ending tool."""
            return "still going"

        agent.register_tool(infinite_tool)
        result = agent.run("go forever")

        assert "Max iterations" in result
        assert "3" in result
        agent.close()


class TestCancellation:
    """Test the cancellation mechanism."""

    def test_cancel_stops_loop(self, mock_env, mock_genai):
        """cancel() should cause the loop to return [Cancelled]."""
        import threading
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        # Always return tool calls — would loop forever without cancellation
        mock_client.models.generate_content.return_value = make_tool_call_response(
            "slow_tool", {}
        )

        agent = Agent()
        agent.MAX_ITERATIONS = 100  # High limit to prove cancellation works

        call_count = 0

        def slow_tool() -> str:
            """A slow tool."""
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                agent.cancel()
            return "done"

        agent.register_tool(slow_tool)
        result = agent.run("go")

        assert result == "[Cancelled]"
        assert call_count < 100  # Did not hit MAX_ITERATIONS
        agent.close()

    def test_cancel_resets_on_new_run(self, mock_env, mock_genai):
        """After a cancelled run, a new run() starts fresh."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        call_count = 0

        # First call: tool call that triggers cancel. Second call: normal text.
        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_tool_call_response("my_tool", {})
            return make_text_response("ok")

        mock_client.models.generate_content.side_effect = side_effect

        agent = Agent()

        def my_tool() -> str:
            """Tool that cancels."""
            agent.cancel()
            return "done"

        agent.register_tool(my_tool)

        # First run gets cancelled mid-loop
        result1 = agent.run("first")
        assert result1 == "[Cancelled]"

        # Reset side_effect for clean second run
        mock_client.models.generate_content.return_value = make_text_response("second ok")
        mock_client.models.generate_content.side_effect = None

        # Second run should complete normally (cancel cleared)
        result2 = agent.run("second")
        assert result2 == "second ok"
        agent.close()

    def test_cancel_from_another_thread(self, mock_env, mock_genai):
        """cancel() called from another thread should stop the loop
        even when a tool is still running."""
        import threading
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        # Model requests two parallel tools
        mock_client.models.generate_content.return_value = make_multi_tool_call_response([
            ("fast_tool", {}),
            ("hung_tool", {}),
        ])

        agent = Agent()
        agent.MAX_ITERATIONS = 100

        fast_done = threading.Event()
        hung_started = threading.Event()

        def fast_tool() -> str:
            """Completes immediately."""
            fast_done.set()
            return "fast result"

        def hung_tool() -> str:
            """Simulates a hung tool — waits until cancelled."""
            hung_started.set()
            # Block for a long time; cancel should break us out
            self_cancel = agent._cancel_event
            self_cancel.wait(timeout=10)
            return "should not matter"

        agent.register_tool(fast_tool)
        agent.register_tool(hung_tool)

        result_holder = {}

        def run_agent():
            result_holder["result"] = agent.run("go")

        t = threading.Thread(target=run_agent)
        t.start()

        # Wait for both tools to be in flight, then cancel
        fast_done.wait(timeout=5)
        hung_started.wait(timeout=5)
        agent.cancel()
        t.join(timeout=10)

        assert result_holder.get("result") == "[Cancelled]"
        agent.close()

    def test_shared_cancel_event_propagates(self, mock_env, mock_genai):
        """Cancelling parent should stop a sub-agent sharing the same event."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_tool_call_response(
            "sub_tool", {}
        )

        parent = Agent()
        child = Agent(cancel_event=parent._cancel_event)

        # They share the same event object
        assert parent._cancel_event is child._cancel_event

        child_calls = 0

        def sub_tool() -> str:
            """Child tool."""
            nonlocal child_calls
            child_calls += 1
            if child_calls >= 2:
                parent.cancel()  # Parent cancels — child should stop too
            return "done"

        child.register_tool(sub_tool)
        result = child.run_stateless("go")

        assert result == "[Cancelled]"
        assert child_calls < 50
        parent.close()
        child.close()

    def test_shared_event_not_cleared_by_sub_agent(self, mock_env, mock_genai):
        """A sub-agent sharing a cancel event must not clear it on run()."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        parent = Agent()
        child = Agent(cancel_event=parent._cancel_event)

        # Parent cancels
        parent.cancel()
        assert parent._cancel_event.is_set()

        # Child starts run_stateless — must NOT clear the shared event
        result = child.run_stateless("go")
        assert result == "[Cancelled]"

        # Event should still be set (parent's cancel is respected)
        assert parent._cancel_event.is_set()
        parent.close()
        child.close()

    def test_own_event_cleared_on_new_run(self, mock_env, mock_genai):
        """An agent with its own cancel event clears it on each run()."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_tool_call_response("t", {})
            return make_text_response("done")

        mock_client.models.generate_content.side_effect = side_effect

        agent = Agent()

        def t() -> str:
            """Tool."""
            agent.cancel()
            return "ok"

        agent.register_tool(t)

        # First run: cancelled mid-loop
        r1 = agent.run("go")
        assert r1 == "[Cancelled]"

        # Second run: event should be cleared, runs to completion
        mock_client.models.generate_content.side_effect = None
        mock_client.models.generate_content.return_value = make_text_response("second")
        r2 = agent.run("again")
        assert r2 == "second"
        agent.close()

    def test_cancel_single_tool_call(self, mock_env, mock_genai):
        """Cancellation should work even with a single tool call."""
        import threading
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_tool_call_response(
            "slow_tool", {}
        )

        agent = Agent()
        agent.MAX_ITERATIONS = 100
        tool_started = threading.Event()

        def slow_tool() -> str:
            """A single slow tool."""
            tool_started.set()
            agent._cancel_event.wait(timeout=10)
            return "done"

        agent.register_tool(slow_tool)

        result_holder = {}

        def run_agent():
            result_holder["result"] = agent.run("go")

        t = threading.Thread(target=run_agent)
        t.start()

        tool_started.wait(timeout=5)
        agent.cancel()
        t.join(timeout=10)

        assert result_holder.get("result") == "[Cancelled]"
        agent.close()

    def test_cancel_does_not_wait_for_noncooperative_tool(self, mock_env, mock_genai):
        """Cancellation should return without waiting for a stuck running tool."""
        from agent_core.agents.base import Agent
        from agent_core.providers.types import ToolCall

        agent = Agent()

        def cancel_tool() -> str:
            """Trigger cancellation."""
            agent.cancel()
            return "cancelled"

        def stuck_tool() -> str:
            """Ignore cancellation for a while."""
            time.sleep(1.2)
            return "late"

        agent.register_tool(cancel_tool)
        agent.register_tool(stuck_tool)

        started = time.perf_counter()
        results = agent._execute_tools_parallel([
            ToolCall(id="call_1", name="cancel_tool", args={}),
            ToolCall(id="call_2", name="stuck_tool", args={}),
        ])
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0
        assert results[1] == ("stuck_tool", {"error": "Cancelled"})
        agent.close()

    def test_cancel_works_with_run_stateless(self, mock_env, mock_genai):
        """cancel() should also work with run_stateless()."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value

        # Always return tool calls
        mock_client.models.generate_content.return_value = make_tool_call_response(
            "my_tool", {}
        )

        agent = Agent()

        def my_tool() -> str:
            """Tool that triggers cancel."""
            agent.cancel()
            return "done"

        agent.register_tool(my_tool)
        result = agent.run_stateless("go")

        assert result == "[Cancelled]"
        agent.close()


class TestAgentAsTool:
    """Test wrapping an agent as a tool for another agent."""

    def test_agent_as_tool_creates_callable(self, mock_env, mock_genai):
        """agent_as_tool should return a callable with correct name."""
        from agent_core.agents.base import Agent, agent_as_tool

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("sub-response")

        sub_agent = Agent()
        sub_agent.name = "researcher"

        tool = agent_as_tool(sub_agent)
        assert tool.__name__ == "researcher_agent"
        assert callable(tool)
        sub_agent.close()

    def test_agent_as_tool_delegates(self, mock_env, mock_genai):
        """Calling the tool should delegate to run_stateless."""
        from agent_core.agents.base import Agent, agent_as_tool

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("delegated")

        sub_agent = Agent()
        tool = agent_as_tool(sub_agent)

        result = tool(task="find papers", context="")
        assert result == "delegated"
        sub_agent.close()


class TestGetContext:
    """Test the debugging/introspection method."""

    def test_get_context_structure(self, mock_env, mock_genai):
        """get_context should return a well-structured dict."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("reply")
        mock_client.models.count_tokens.return_value = MagicMock(total_tokens=500)

        agent = Agent()
        agent.run("hello")

        ctx = agent.get_context()
        assert "system_prompt" in ctx
        assert "history" in ctx
        assert "tools" in ctx
        assert "agent_type" in ctx
        assert "instance_id" in ctx
        assert "context_tokens" in ctx
        assert len(ctx["history"]) > 0
        agent.close()


class TestSummarizeResult:
    """Test the _summarize_result method."""

    def test_dict_with_error(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        agent = Agent()
        assert agent._summarize_result({"error": "boom"}) == "Error: boom"
        agent.close()

    def test_dict_with_text(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        agent = Agent()
        assert agent._summarize_result({"text": "hello"}) == "hello"
        agent.close()

    def test_dict_with_result_key(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        agent = Agent()
        assert agent._summarize_result({"result": "value"}) == "value"
        agent.close()

    def test_dict_fallback_shows_keys(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        agent = Agent()
        summary = agent._summarize_result({"a": 1, "b": 2})
        assert "a" in summary and "b" in summary
        agent.close()

    def test_string_truncation(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        agent = Agent()
        long_str = "x" * 300
        summary = agent._summarize_result(long_str)
        assert len(summary) < 300
        assert summary.endswith("...")
        agent.close()

    def test_list_result(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        agent = Agent()
        assert agent._summarize_result([1, 2, 3]) == "List[3 items]"
        agent.close()

    def test_none_result(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        agent = Agent()
        assert agent._summarize_result(None) == "None"
        agent.close()

    def test_overridable_in_subclass(self, mock_env, mock_genai):
        """Subclass can override _summarize_result for domain-specific logic."""
        from agent_core.agents.base import Agent

        class CustomAgent(Agent):
            def _summarize_result(self, result, max_length=200):
                if isinstance(result, dict) and "custom_key" in result:
                    return f"Custom: {result['custom_key']}"
                return super()._summarize_result(result, max_length)

        agent = CustomAgent()
        assert agent._summarize_result({"custom_key": "val"}) == "Custom: val"
        assert agent._summarize_result({"text": "hello"}) == "hello"
        agent.close()


class TestEventEmission:
    """Test that events are emitted at the right lifecycle points."""

    def test_agent_start_event_emitted(self, mock_env, mock_genai):
        """AGENT_START event should be emitted when run() is called."""
        from agent_core.agents.base import Agent
        from agent_core.core.events import EventBus, EventType

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        bus = EventBus()
        agent = Agent(event_bus=bus)
        agent.run("test")

        start_events = [e for e in bus.get_events() if e.type == EventType.AGENT_START]
        assert len(start_events) == 1
        agent.close()

    def test_agent_end_event_emitted(self, mock_env, mock_genai):
        """AGENT_END event should be emitted when run() completes."""
        from agent_core.agents.base import Agent
        from agent_core.core.events import EventBus, EventType

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        bus = EventBus()
        agent = Agent(event_bus=bus)
        agent.run("test")

        end_events = [e for e in bus.get_events() if e.type == EventType.AGENT_END]
        assert len(end_events) == 1
        agent.close()

    def test_events_disabled(self, mock_env, mock_genai):
        """Events should not be emitted when disabled."""
        from agent_core.agents.base import Agent
        from agent_core.core.events import EventBus

        class SilentAgent(Agent):
            emit_lifecycle_events = False
            emit_tool_events = False

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        bus = EventBus()
        agent = SilentAgent(event_bus=bus)
        agent.run("test")

        assert len(bus.get_events()) == 0
        agent.close()

    def test_isolated_event_bus(self, mock_env, mock_genai):
        """Agents with different event buses should not see each other's events."""
        from agent_core.agents.base import Agent
        from agent_core.core.events import EventBus

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        bus_a = EventBus()
        bus_b = EventBus()
        agent_a = Agent(event_bus=bus_a)
        agent_b = Agent(event_bus=bus_b)

        agent_a.run("hello")

        assert len(bus_a.get_events()) > 0
        assert len(bus_b.get_events()) == 0
        agent_a.close()
        agent_b.close()


class TestCompaction:
    """Test automatic context compaction in the shared runtime."""

    def test_enabled_default_compaction_config_is_safe(self, mock_env, mock_genai):
        """ENABLE_COMPACTION=True should not trigger zero-token summaries."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        class DefaultCompactingAgent(Agent):
            ENABLE_COMPACTION = True

        agent = DefaultCompactingAgent()
        agent._history.extend([
            agent._provider.build_user_message("u1"),
            MockContent(role="model", parts=[MockPart(text="m1")]),
            agent._provider.build_user_message("u2"),
            MockContent(role="model", parts=[MockPart(text="m2")]),
        ])

        result = agent.run("latest")

        assert result == "ok"
        assert mock_client.models.generate_content.call_count == 1
        agent.close()

    def test_run_compacts_history_before_model_call(self, mock_env, mock_genai):
        """run() should compact old history when the context threshold is exceeded."""
        from agent_core.agents.base import Agent
        from agent_core.agents.compaction import CompactionConfig

        mock_client = mock_genai.Client.return_value

        token_counts = iter([MockTokenCountResponse(80), MockTokenCountResponse(20), MockTokenCountResponse(20)])
        mock_client.models.count_tokens.side_effect = lambda *args, **kwargs: next(token_counts)
        mock_client.models.generate_content.side_effect = [
            make_text_response("Compacted summary"),
            make_text_response("done"),
        ]

        hook_calls = []

        class CompactingAgent(Agent):
            ENABLE_COMPACTION = True

            def get_compaction_config(self):
                return CompactionConfig(
                    enabled=True,
                    model_limit_tokens=100,
                    trigger_tokens=50,
                    target_tokens=30,
                    tail_token_budget=1,
                    response_buffer_tokens=20,
                    summary_max_output_tokens=128,
                    max_transcript_chars=4000,
                    max_message_chars=1000,
                    min_preserved_messages=1,
                    max_compactions_per_run=2,
                )

            def on_compaction_start(self, **kwargs):
                hook_calls.append(("start", kwargs["scope"], kwargs["pre_tokens"]))

            def on_compaction_complete(self, **kwargs):
                hook_calls.append(
                    ("complete", kwargs["scope"], kwargs["pre_tokens"], kwargs["post_tokens"])
                )

        agent = CompactingAgent()
        agent._history.append(agent._provider.build_user_message("Earlier context"))

        result = agent.run("Latest request")

        assert result == "done"
        assert hook_calls == [
            ("start", "session", 80),
            ("complete", "session", 80, 20),
        ]
        assert mock_client.models.generate_content.call_count == 2
        summary_display = agent._provider.format_message_for_display(agent._history[0])
        assert summary_display is not None
        assert "Internal context compaction summary" in summary_display["content"]
        agent.close()

    def test_run_stateless_can_compact_mid_run(self, mock_env, mock_genai):
        """run_stateless() should compact accumulated context for sub-agent style runs."""
        from agent_core.agents.base import Agent
        from agent_core.agents.compaction import CompactionConfig

        mock_client = mock_genai.Client.return_value

        token_counts = iter([
            MockTokenCountResponse(90),
            MockTokenCountResponse(25),
            MockTokenCountResponse(25),
        ])
        mock_client.models.count_tokens.side_effect = lambda *args, **kwargs: next(token_counts)
        mock_client.models.generate_content.side_effect = [
            make_tool_call_response("my_tool", {}),
            make_text_response("Stateless compacted summary"),
            make_text_response("done"),
        ]

        hook_calls = []

        class CompactingAgent(Agent):
            ENABLE_COMPACTION = True

            def get_compaction_config(self):
                return CompactionConfig(
                    enabled=True,
                    model_limit_tokens=100,
                    trigger_tokens=50,
                    target_tokens=30,
                    tail_token_budget=1,
                    response_buffer_tokens=20,
                    summary_max_output_tokens=128,
                    max_transcript_chars=4000,
                    max_message_chars=1000,
                    min_preserved_messages=1,
                    max_compactions_per_run=2,
                )

            def on_compaction_start(self, **kwargs):
                hook_calls.append(("start", kwargs["scope"], kwargs["pre_tokens"]))

            def on_compaction_complete(self, **kwargs):
                hook_calls.append(
                    ("complete", kwargs["scope"], kwargs["pre_tokens"], kwargs["post_tokens"])
                )

        agent = CompactingAgent()

        def my_tool() -> str:
            """Simple tool used to grow stateless history."""
            return "tool output"

        agent.register_tool(my_tool)
        result = agent.run_stateless("Need tool work first")

        assert result == "done"
        assert hook_calls == [
            ("start", "run", 90),
            ("complete", "run", 90, 25),
        ]
        assert mock_client.models.generate_content.call_count == 3
        agent.close()
