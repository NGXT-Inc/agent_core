"""Tests for the Agent base class (excluding caching, tested in test_caching.py)."""

import os
from unittest.mock import MagicMock, patch, call

import pytest

from tests.conftest import (
    MockContent,
    MockPart,
    MockFunctionCall,
    MockResponse,
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

        with patch.dict(os.environ, {}, clear=True), \
             patch("agent_core.agents.base.GOOGLE_PROJECT_ID", None):
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
            def on_agent_end(self, result, success, error=None):
                end_args["result"] = result
                end_args["success"] = success
                end_args["error"] = error

        agent = HookedAgent()
        agent.run("test")

        assert end_args["success"] is True
        assert end_args["result"] == "ok"
        assert end_args["error"] is None
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


class TestEventEmission:
    """Test that events are emitted at the right lifecycle points."""

    def test_agent_start_event_emitted(self, mock_env, mock_genai, mock_events):
        """AGENT_START event should be emitted when run() is called."""
        from agent_core.agents.base import Agent
        from agent_core.core.events import EventType

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = Agent()
        agent.run("test")

        # Find the AGENT_START call
        start_calls = [
            c for c in mock_events.call_args_list
            if c.args and c.args[0] == EventType.AGENT_START
        ]
        assert len(start_calls) == 1
        agent.close()

    def test_agent_end_event_emitted(self, mock_env, mock_genai, mock_events):
        """AGENT_END event should be emitted when run() completes."""
        from agent_core.agents.base import Agent
        from agent_core.core.events import EventType

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = Agent()
        agent.run("test")

        end_calls = [
            c for c in mock_events.call_args_list
            if c.args and c.args[0] == EventType.AGENT_END
        ]
        assert len(end_calls) == 1
        agent.close()

    def test_events_disabled(self, mock_env, mock_genai, mock_events):
        """Events should not be emitted when disabled."""
        from agent_core.agents.base import Agent

        class SilentAgent(Agent):
            emit_lifecycle_events = False
            emit_tool_events = False

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        agent = SilentAgent()
        agent.run("test")

        mock_events.assert_not_called()
        agent.close()
