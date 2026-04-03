"""Compatibility tests for all Papyrus downstream usage patterns.

Tests every import pattern, __init__ signature, method override, internal
attribute access, and utility function used by the 16+ files in the
Papyrus backend that depend on agent_core.

These tests use mocks — no Gemini or OpenRouter API calls.
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    MockContent,
    MockFunctionCall,
    MockPart,
    MockResponse,
    MockUsageMetadata,
    make_text_response,
    make_tool_call_response,
)


# ============================================================
# 1. Import compatibility — every import pattern used by Papyrus
# ============================================================


class TestImportCompat:
    """Verify all Papyrus import statements resolve without error."""

    def test_agent_base_imports(self):
        """agent_base.py: Agent, agent_as_tool, MODEL_PRO, MODEL_FLASH, etc."""
        from agent_core.agents.base import (
            Agent,
            agent_as_tool,
            generate_instance_id,
            MODEL_PRO,
            MODEL_FLASH,
        )
        assert MODEL_PRO is not None
        assert MODEL_FLASH is not None

    def test_lazy_env_imports(self):
        """generate_image.py, github_link.py: GOOGLE_PROJECT_ID, GOOGLE_LOCATION."""
        from agent_core.agents.base import GOOGLE_PROJECT_ID, GOOGLE_LOCATION
        # These are lazy — just verify they don't raise AttributeError
        assert GOOGLE_LOCATION is not None  # defaults to "global"

    def test_persistence_imports(self):
        """chat_persistence.py, session_manager.py, general_session_manager.py."""
        from agent_core.core.persistence import (
            ConversationStoreProtocol,
            InMemoryConversationStore,
            SQLiteConversationStore,
            serialize_content,
            deserialize_content,
        )

    def test_events_imports(self):
        """general.py, researcher.py, routes.py, sdk_routes.py."""
        from agent_core.core.events import emit_event, EventType, EventStatus
        from agent_core import get_event_bus

    def test_provider_imports(self):
        """New provider types are importable."""
        from agent_core import GeminiProvider, LLMProvider, ToolCall, ParsedResponse
        from agent_core.providers.openai import OpenAIProvider


# ============================================================
# 2. Agent __init__ signatures — every Papyrus subclass pattern
# ============================================================


class TestInitSignatures:
    """Verify every Papyrus __init__ pattern works with new Agent."""

    def test_general_agent_pattern(self, mock_env, mock_genai):
        """GeneralAgent: super().__init__(model_name=..., session_id=..., conversation_store=...)"""
        from agent_core.agents.base import Agent, MODEL_PRO
        from agent_core.core.persistence import InMemoryConversationStore

        store = InMemoryConversationStore()

        class GeneralAgent(Agent):
            name = "general"

        agent = GeneralAgent(
            model_name=MODEL_PRO,
            session_id="test-session",
            conversation_store=store,
        )
        assert agent.model_name == MODEL_PRO
        assert agent._session_id == "test-session"
        agent.close()

    def test_explorer_agent_pattern(self, mock_env, mock_genai):
        """ExplorerAgent: super().__init__(model_name=..., parent_agent=...)"""
        from agent_core.agents.base import Agent, MODEL_FLASH

        class ExplorerAgent(Agent):
            name = "explorer"

        agent = ExplorerAgent(
            model_name=MODEL_FLASH,
            parent_agent="general_abc12345",
        )
        assert agent._parent_agent == "general_abc12345"
        agent.close()

    def test_sandbox_agent_pattern(self, mock_env, mock_genai):
        """SandboxAgent: super().__init__(model_name=..., parent_agent=...)"""
        from agent_core.agents.base import Agent, MODEL_FLASH

        class SandboxAgent(Agent):
            name = "sandbox_manager"

        agent = SandboxAgent(
            model_name=MODEL_FLASH,
            parent_agent="general_abc12345",
        )
        agent.close()

    def test_agent_base_wrapper_pattern(self, mock_env, mock_genai):
        """agent_base.py: super().__init__(model_name, parent_agent, session_id, conversation_store)"""
        from agent_core.agents.base import Agent

        class LDIAAgent(Agent):
            name = "ldia_base"

            def __init__(self, model_name=None, parent_agent=None,
                         session_id=None, conversation_store=None):
                super().__init__(
                    model_name=model_name,
                    parent_agent=parent_agent,
                    session_id=session_id,
                    conversation_store=conversation_store,
                )

        agent = LDIAAgent(model_name="test-model", session_id="s1")
        agent.close()

    def test_injected_client_pattern(self, mock_client):
        """Test backward-compat client injection (used in tests)."""
        from agent_core.agents.base import Agent

        agent = Agent(client=mock_client)
        assert agent.client is mock_client
        agent.close()

    def test_provider_injection_pattern(self):
        """New pattern: inject a custom provider."""
        from agent_core.agents.base import Agent
        from agent_core.providers.openai import OpenAIProvider

        mock_openai_client = MagicMock()
        provider = OpenAIProvider(client=mock_openai_client)
        agent = Agent(provider=provider, model_name="gpt-4o")
        assert agent.model_name == "gpt-4o"
        assert agent._provider is provider
        agent.close()


# ============================================================
# 3. ROOT_AGENT_TYPES — Papyrus uses .add()
# ============================================================


class TestRootAgentTypes:
    """Test ROOT_AGENT_TYPES compatibility."""

    def test_root_agent_types_is_frozenset(self):
        """ROOT_AGENT_TYPES was changed to frozenset for immutability."""
        from agent_core.agents.base import Agent
        assert isinstance(Agent.ROOT_AGENT_TYPES, frozenset)

    def test_add_raises_on_frozenset(self):
        """Papyrus calls .add() — this WILL fail on frozenset."""
        from agent_core.agents.base import Agent
        with pytest.raises(AttributeError):
            Agent.ROOT_AGENT_TYPES.add("researcher")

    def test_subclass_override_pattern(self, mock_env, mock_genai):
        """The recommended pattern: override ROOT_AGENT_TYPES in subclass."""
        from agent_core.agents.base import Agent

        class ResearcherAgent(Agent):
            name = "researcher"
            ROOT_AGENT_TYPES = frozenset({"researcher", *Agent.ROOT_AGENT_TYPES})

        agent = ResearcherAgent(session_id="abcdef1234567890")
        # Should get deterministic ID since "researcher" is in ROOT_AGENT_TYPES
        assert agent.instance_id == "researcher_abcdef12"
        agent.close()


# ============================================================
# 4. agent_as_tool — GeneralAgent wraps sub-agents
# ============================================================


class TestAgentAsTool:
    """Test agent_as_tool wrapping pattern used by GeneralAgent."""

    def test_wrap_and_register(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent, agent_as_tool, MODEL_FLASH

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("sub-result")

        class ParentAgent(Agent):
            name = "parent"

        class ChildAgent(Agent):
            name = "explorer"

        parent = ParentAgent()
        child = ChildAgent(parent_agent=parent.instance_id)

        tool = agent_as_tool(child, description="Explore things")
        parent.register_tool(tool)

        assert "explorer_agent" in parent._tools
        parent.close()
        child.close()


# ============================================================
# 5. Event emission — GeneralAgent & ResearcherAgent use emit_event()
# ============================================================


class TestEventEmission:
    """Test event patterns used by Papyrus agents."""

    def test_emit_event_function(self):
        """emit_event() called directly from on_tool_end hooks."""
        from agent_core.core.events import emit_event, EventType, EventStatus

        event = emit_event(
            event_type="papers_added",
            agent="researcher_abc12345",
            agent_type="researcher",
            status=EventStatus.COMPLETED,
            details={"papers": ["paper1", "paper2"]},
        )
        assert event.type == "papers_added"
        assert event.details["papers"] == ["paper1", "paper2"]

    def test_get_event_bus_subscribe(self):
        """routes.py: get_event_bus().subscribe(listener)"""
        from agent_core import get_event_bus

        received = []
        def listener(event):
            received.append(event)

        bus = get_event_bus()
        bus.subscribe(listener)

        from agent_core.core.events import emit_event
        emit_event("test_event", agent="test")

        assert len(received) >= 1
        bus.unsubscribe(listener)


# ============================================================
# 6. Serialization — chat_persistence.py, general_session_manager.py
# ============================================================


class TestSerialization:
    """Test Gemini-specific serialization still works (used by Papyrus directly)."""

    def test_serialize_content_preserved(self):
        """chat_persistence.py calls serialize_content(c) directly."""
        from agent_core.core.persistence import serialize_content

        content = MockContent(
            role="user",
            parts=[MockPart(text="hello")],
        )
        result = serialize_content(content)
        assert result["role"] == "user"
        assert result["parts"][0]["type"] == "text"
        assert result["parts"][0]["text"] == "hello"

    def test_deserialize_content_preserved(self):
        """general_session_manager.py calls deserialize_content(d) directly."""
        from agent_core.core.persistence import deserialize_content

        data = {
            "role": "model",
            "parts": [{"type": "text", "text": "hello"}],
        }
        # This calls the real deserialize which uses google.genai types
        # We just verify it doesn't raise
        content = deserialize_content(data)
        assert content.role == "model"

    def test_new_dispatch_helpers_exist(self):
        """New serialize_message/deserialize_message dispatch helpers."""
        from agent_core.core.persistence import serialize_message, deserialize_message

        # With provider=None, should fall back to Gemini path
        content = MockContent(role="user", parts=[MockPart(text="test")])
        result = serialize_message(content, provider=None)
        assert result["role"] == "user"


# ============================================================
# 7. Cache registry — server.py uses class methods
# ============================================================


class TestCacheRegistry:
    """Test cache registry class methods used by server.py."""

    def test_init_and_shutdown_cache_registry(self):
        """server.py: Agent.init_cache_registry(client, ...) / shutdown()"""
        from agent_core.agents.base import Agent

        mock_client = MagicMock()
        Agent.init_cache_registry(mock_client, max_workers=2, cache_ttl_seconds=300)
        assert Agent._cache_registry is not None

        Agent.shutdown_cache_registry()
        assert Agent._cache_registry is None


# ============================================================
# 8. Internal attribute access — check_agent_tools.py
# ============================================================


class TestInternalAttributeAccess:
    """Test internal attribute access patterns from check_agent_tools.py."""

    def test_tools_dict_accessible(self, mock_env, mock_genai):
        """check_agent_tools.py: list(agent._tools.keys())"""
        from agent_core.agents.base import Agent

        agent = Agent()

        def my_tool(query: str) -> str:
            """A tool."""
            return "ok"

        agent.register_tool(my_tool)
        assert "my_tool" in agent._tools
        agent.close()

    def test_execute_tools_parallel_accessible(self, mock_env, mock_genai):
        """check_agent_tools.py: agent._execute_tools_parallel monkey-patch"""
        from agent_core.agents.base import Agent
        from agent_core.providers.types import ToolCall

        agent = Agent()
        # Verify the method exists and is callable
        assert callable(agent._execute_tools_parallel)

        # Verify it accepts ToolCall objects (new signature)
        def my_tool(query: str) -> str:
            """A tool."""
            return "result"

        agent.register_tool(my_tool)

        tc = ToolCall(id="tc_1", name="my_tool", args={"query": "test"})
        results = agent._execute_tools_parallel([tc])
        assert len(results) == 1
        assert results[0][0] == "my_tool"
        assert results[0][1] == "result"
        agent.close()

    def test_on_tool_hooks_overridable(self, mock_env, mock_genai):
        """check_agent_tools.py: agent.on_tool_start = tracking_fn"""
        from agent_core.agents.base import Agent

        agent = Agent()
        calls = []

        original = agent.on_tool_start
        agent.on_tool_start = lambda name, args, tid: calls.append(name)

        # Verify assignment works
        agent.on_tool_start("test", {}, "id_1")
        assert calls == ["test"]
        agent.close()


# ============================================================
# 9. Method overrides — SandboxAgent.run_stateless, AuthorExtraction._run_with_function_loop
# ============================================================


class TestMethodOverrides:
    """Test that downstream method overrides still work."""

    def test_run_stateless_override(self, mock_env, mock_genai):
        """SandboxAgent overrides run_stateless to inject context."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("ok")

        class MySandboxAgent(Agent):
            name = "sandbox_manager"

            def run_stateless(self, prompt, context=None, temperature=0.7, max_output_tokens=32768):
                enriched = f"[sandbox status]\n{prompt}"
                return super().run_stateless(
                    enriched, context=context,
                    temperature=temperature, max_output_tokens=max_output_tokens,
                )

        agent = MySandboxAgent()
        result = agent.run_stateless("do something")
        assert result == "ok"
        agent.close()

    def test_run_with_function_loop_override(self, mock_env, mock_genai):
        """AuthorExtractionAgent overrides _run_with_function_loop to capture usage."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("done")

        class MyAgent(Agent):
            name = "author_extractor"
            last_usage = None

            def _run_with_function_loop(self, contents, temperature, max_output_tokens, **kwargs):
                result, usage = super()._run_with_function_loop(
                    contents, temperature, max_output_tokens, **kwargs
                )
                self.last_usage = usage
                return result, usage

        agent = MyAgent()
        agent.run_stateless("test")
        assert agent.last_usage is not None
        assert "prompt_tokens" in agent.last_usage
        agent.close()


# ============================================================
# 10. Lifecycle hooks with tool calling
# ============================================================


class TestLifecycleHooksWithTools:
    """Test on_tool_start/on_tool_end receive correct args (GeneralAgent, ResearcherAgent)."""

    def test_hooks_called_with_correct_args(self, mock_env, mock_genai):
        """GeneralAgent uses on_tool_end(tool_name, args, tool_call_id, result, success, error)."""
        from agent_core.agents.base import Agent

        mock_client = mock_genai.Client.return_value
        responses = [
            make_tool_call_response("my_tool", {"query": "test"}),
            make_text_response("done"),
        ]
        mock_client.models.generate_content.side_effect = responses

        hook_calls = []

        class HookedAgent(Agent):
            name = "hooked"

            def on_tool_start(self, tool_name, args, tool_call_id):
                hook_calls.append(("start", tool_name, args, tool_call_id))

            def on_tool_end(self, tool_name, args, tool_call_id, result, success, error=None):
                hook_calls.append(("end", tool_name, success, result))

        agent = HookedAgent()

        def my_tool(query: str) -> str:
            """A tool."""
            return "found it"

        agent.register_tool(my_tool)
        agent.run_stateless("go")

        assert len(hook_calls) == 2
        assert hook_calls[0][0] == "start"
        assert hook_calls[0][1] == "my_tool"
        assert hook_calls[1][0] == "end"
        assert hook_calls[1][1] == "my_tool"
        assert hook_calls[1][2] is True  # success
        assert hook_calls[1][3] == "found it"  # result
        agent.close()


# ============================================================
# 11. Multi-turn stateful conversation with persistence
# ============================================================


class TestStatefulConversation:
    """Test multi-turn run() with InMemoryConversationStore."""

    def test_history_persists_across_turns(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent
        from agent_core.core.persistence import InMemoryConversationStore

        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = make_text_response("reply")

        store = InMemoryConversationStore()
        agent = Agent(session_id="test-session", conversation_store=store)
        agent.run("msg1")
        agent.run("msg2")

        saved = store.load("test-session", "base")
        assert len(saved) == 4  # user1, model1, user2, model2
        agent.close()


# ============================================================
# 12. Cancellation propagation (parent -> child via shared event)
# ============================================================


class TestCancellationPropagation:
    """Test cancel_event sharing (GeneralAgent -> sub-agents)."""

    def test_shared_cancel_event(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        parent = Agent()
        child = Agent(cancel_event=parent._cancel_event)

        parent.cancel()
        assert child._cancel_event.is_set()
        parent.close()
        child.close()


# ============================================================
# 13. Class attribute overrides
# ============================================================


class TestClassAttributes:
    """Test that all class attribute overrides used by Papyrus work."""

    def test_all_overridable_attrs(self, mock_env, mock_genai):
        from agent_core.agents.base import Agent

        class CustomAgent(Agent):
            name = "custom"
            system_prompt = "You are custom."
            MAX_ITERATIONS = 10
            MAX_PARALLEL_TOOLS = 1
            CODE_TOOLS: set[str] = set()
            ENABLE_CACHING = False
            CACHE_MIN_TOKENS = 16_000
            emit_lifecycle_events = False
            emit_tool_events = False

        agent = CustomAgent()
        assert agent.MAX_ITERATIONS == 10
        assert agent.MAX_PARALLEL_TOOLS == 1
        assert agent.ENABLE_CACHING is False
        assert agent.emit_lifecycle_events is False
        agent.close()
