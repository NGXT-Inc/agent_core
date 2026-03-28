"""Base agent class with Gemini function-calling support.

Each agent is a Gemini model instance with:
- A specific system prompt defining its role
- A set of tools (Python functions) it can call
- Manual function calling loop with parallel execution support

This module is designed to be extended without modification:
- Override class attributes for configuration
- Override hook methods for custom behavior
- Inject custom ConversationStore for persistence
"""

import functools
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, ClassVar

logger = logging.getLogger(__name__)

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types

from agent_core.core.caching import CachePipeline
from agent_core.core.events import EventType, EventStatus, emit_event
from agent_core.core.persistence import ConversationStoreProtocol, InMemoryConversationStore

load_dotenv(find_dotenv(usecwd=True))

# Default model constants (can be overridden at class level)
MODEL_PRO = "gemini-3.1-pro-preview"
MODEL_FLASH = "gemini-3-flash-preview"

# Environment configuration
GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
GOOGLE_LOCATION = os.environ.get("GOOGLE_LOCATION", "global")


def generate_instance_id(
    agent_type: str,
    session_id: str | None = None,
    root_agent_types: set[str] | None = None,
) -> str:
    """Generate a unique instance ID for an agent invocation.

    Args:
        agent_type: The type of agent (e.g., "designer", "implementer").
        session_id: Optional session ID. If provided and agent is a root type,
                   uses first 8 chars for deterministic ID.
        root_agent_types: Set of agent types considered "root" agents.
                         Defaults to Agent.ROOT_AGENT_TYPES.

    Returns:
        Unique ID like "designer_abc123".
    """
    if root_agent_types is None:
        root_agent_types = Agent.ROOT_AGENT_TYPES

    if session_id and agent_type in root_agent_types:
        # Use session-based ID for deterministic identification
        # This allows frontend and backend to generate matching IDs
        short_id = session_id[:8]
    else:
        # Use random UUID for sub-agents (multiple instances per session)
        short_id = uuid.uuid4().hex[:8]
    return f"{agent_type}_{short_id}"


class Agent:
    """Base class for all agents.

    Agents are specialized Gemini instances with specific prompts and tools.
    Uses manual function calling loop with parallel execution support.

    Class Attributes (override in subclasses):
        name: Agent type identifier.
        system_prompt: Default system prompt.
        DEFAULT_MODEL: Model to use if not specified.
        ROOT_AGENT_TYPES: Agent types that get deterministic IDs from session.
        CODE_TOOLS: Tool names that handle code (for special result formatting).
        MAX_PARALLEL_TOOLS: Maximum concurrent tool executions.
        MAX_ITERATIONS: Maximum function calling loop iterations.
        emit_lifecycle_events: Whether to emit AGENT_START/END events.
        emit_tool_events: Whether to emit TOOL_START/END events.

    Usage:
        class ResearcherAgent(Agent):
            name = "researcher"
            system_prompt = "You are a literature research assistant..."

            def __init__(self, session_id: str = None):
                super().__init__(session_id=session_id)
                self.register_tool(self.search_papers)
                self.register_tool(self.read_paper)

            def search_papers(self, query: str) -> list[dict]:
                '''Search for research papers.'''
                ...
    """

    # --- Class-level configuration (override in subclasses) ---

    name: str = "base"
    system_prompt: str = "You are a helpful assistant."

    # Default model (override per agent type)
    DEFAULT_MODEL: ClassVar[str] = MODEL_PRO

    # Agent types that get deterministic instance IDs from session_id
    # Add your root agent types here: Agent.ROOT_AGENT_TYPES.add("my_agent")
    ROOT_AGENT_TYPES: ClassVar[set[str]] = {"designer", "analyst", "data_analyst"}

    # Tools that handle code execution (for special result formatting in events)
    # Override or extend in subclasses as needed
    CODE_TOOLS: ClassVar[set[str]] = {
        "execute_code",
        "write_and_execute_block",
        "update_and_execute_block",
    }

    # Execution limits
    MAX_PARALLEL_TOOLS: ClassVar[int] = 10
    MAX_ITERATIONS: ClassVar[int] = 50

    # Context caching (pipelined, transparent)
    ENABLE_CACHING: ClassVar[bool] = True
    CACHE_MIN_TOKENS: ClassVar[int] = 32_768

    # Event emission flags (set False to handle events yourself)
    emit_lifecycle_events: ClassVar[bool] = True
    emit_tool_events: ClassVar[bool] = True

    def __init__(
        self,
        model_name: str | None = None,
        parent_agent: str | None = None,
        session_id: str | None = None,
        conversation_store: ConversationStoreProtocol | None = None,
    ):
        """Initialize the agent.

        Args:
            model_name: Model to use. Defaults to class DEFAULT_MODEL.
            parent_agent: Instance ID of the parent agent (for graph visualization).
            session_id: Optional session ID for deterministic instance ID generation
                       and conversation persistence.
            conversation_store: Optional persistence backend. If None and session_id
                              is provided, uses InMemoryConversationStore.
        """
        if not GOOGLE_PROJECT_ID:
            raise ValueError("GOOGLE_PROJECT_ID must be set in environment or .env file")

        self.model_name = model_name or self.DEFAULT_MODEL

        # Generate unique instance ID
        self.instance_id = generate_instance_id(self.name, session_id, self.ROOT_AGENT_TYPES)

        # Store session_id for persistence
        self._session_id: str | None = session_id

        # Initialize Gemini client
        self.client = genai.Client(
            vertexai=True, project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION
        )

        # Tools registry - maps function name to callable
        self._tools: dict[str, Callable] = {}

        # Tool list for SDK (function declarations)
        self._tool_functions: list[Callable] = []

        # Conversation persistence
        self._conversation_store = conversation_store
        if session_id and self._conversation_store:
            self._history: list[types.Content] = self._conversation_store.load(
                session_id, self.name
            )
        else:
            self._history: list[types.Content] = []

        # Parent agent instance ID for event tracking (graph edges)
        self._parent_agent: str | None = parent_agent

        # Wave ID for grouping parallel agents in visualization
        self._wave_id: str | None = None

        # Context caching pipeline (only for persistent sessions —
        # avoids overhead on short-lived sub-agents using run_stateless)
        if self.ENABLE_CACHING and session_id:
            self._cache_pipeline = CachePipeline(
                client=self.client,
                model_name=self.model_name,
                min_token_threshold=self.CACHE_MIN_TOKENS,
            )
            self._cache_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="cache"
            )
        else:
            self._cache_pipeline = None
            self._cache_executor = None

    # --- Lifecycle Hooks (override in subclasses) ---

    def on_agent_start(self, prompt: str) -> None:
        """Called when agent starts processing a prompt.

        Override to add custom logging, metrics, or setup.

        Args:
            prompt: The user prompt being processed.
        """
        pass

    def on_agent_end(self, result: str, success: bool, error: str | None = None) -> None:
        """Called when agent finishes processing.

        Override to add custom logging, cleanup, or metrics.

        Args:
            result: The final result text (empty if failed).
            success: Whether processing completed successfully.
            error: Error message if failed.
        """
        pass

    def on_tool_start(self, tool_name: str, args: dict, tool_call_id: str) -> None:
        """Called before a tool is executed.

        Override to add custom logging, validation, or preprocessing.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments passed to the tool.
            tool_call_id: Unique ID for this tool invocation.
        """
        pass

    def on_tool_end(
        self,
        tool_name: str,
        args: dict,
        tool_call_id: str,
        result: Any,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Called after a tool finishes executing.

        Override to add custom logging, caching, or postprocessing.

        Args:
            tool_name: Name of the tool that was called.
            args: Arguments that were passed to the tool.
            tool_call_id: Unique ID for this tool invocation.
            result: The tool's return value (None if failed).
            success: Whether the tool executed successfully.
            error: Error message if failed.
        """
        pass

    def on_model_thinking(self, text: str) -> None:
        """Called when model emits intermediate reasoning text.

        Override to capture or display model's thinking process.

        Args:
            text: The intermediate text from the model.
        """
        pass

    # --- Tool Registration ---

    def register_tool(self, func: Callable) -> None:
        """Register a tool function for this agent.

        The function should have type hints and a docstring.
        The SDK will automatically convert it to a function declaration.

        Args:
            func: A callable with type hints and docstring.
        """
        tool_name = getattr(func, "__name__", str(func))
        # Store both the wrapped version (for execution) and original (for SDK)
        wrapped = self._wrap_tool_with_events(func)
        self._tools[tool_name] = wrapped
        self._tool_functions.append(func)  # SDK needs unwrapped for declarations
        if self._cache_pipeline:
            self._cache_pipeline.invalidate()

    def _wrap_tool_with_events(self, func: Callable) -> Callable:
        """Wrap a tool function to emit events and call hooks."""
        tool_name = getattr(func, "__name__", str(func))
        # Clean up tool name (remove leading underscore if present)
        display_name = tool_name.lstrip("_")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate unique tool_call_id for pairing START with END
            tool_call_id = f"{display_name}_{uuid.uuid4().hex[:8]}"

            # Build args hint for visualization
            args_hint = None
            full_code = None

            if kwargs:
                for key, val in kwargs.items():
                    if isinstance(val, str) and val:
                        hint_val = val[:30] + "..." if len(val) > 30 else val
                        args_hint = hint_val
                        # For code tools, store full code
                        if display_name in self.CODE_TOOLS and key == "code":
                            full_code = val
                        break

            details = {"args_hint": args_hint} if args_hint else {}
            if full_code:
                details["code"] = full_code
            if "cell_index" in kwargs:
                details["cell_index"] = kwargs["cell_index"]

            # Call hook
            self.on_tool_start(display_name, kwargs, tool_call_id)

            # Emit event if enabled
            if self.emit_tool_events:
                emit_event(
                    EventType.TOOL_START,
                    agent=self.instance_id,
                    agent_type=self.name,
                    tool=display_name,
                    tool_call_id=tool_call_id,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details=details,
                )

            try:
                result = func(*args, **kwargs)

                # Summarize result for UI
                result_summary = self._summarize_result(result)
                result_type = type(result).__name__

                result_details = {
                    **details,
                    "result_summary": result_summary,
                    "result_type": result_type,
                }

                # For code tools, include full output details
                if display_name in self.CODE_TOOLS and isinstance(result, dict):
                    code_result_keys = [
                        "code",
                        "stdout",
                        "stderr",
                        "success",
                        "valid",
                        "error",
                        "rich_outputs",
                        "defined_variables",
                        "cell_index",
                        "description",
                    ]
                    for key in code_result_keys:
                        if key in result:
                            result_details[key] = result[key]

                    result_details["cell_name"] = (
                        result.get("cell_name")
                        or result.get("notebook_cell_name")
                        or f"cell_{tool_call_id}"
                    )

                # Call hook
                self.on_tool_end(display_name, kwargs, tool_call_id, result, success=True)

                # Emit event if enabled
                if self.emit_tool_events:
                    emit_event(
                        EventType.TOOL_END,
                        agent=self.instance_id,
                        agent_type=self.name,
                        tool=display_name,
                        tool_call_id=tool_call_id,
                        status=EventStatus.COMPLETED,
                        parent_agent=self._parent_agent,
                        wave_id=self._wave_id,
                        details=result_details,
                    )
                return result

            except Exception as e:
                # Call hook
                self.on_tool_end(
                    display_name, kwargs, tool_call_id, None, success=False, error=str(e)
                )

                # Emit event if enabled
                if self.emit_tool_events:
                    emit_event(
                        EventType.TOOL_END,
                        agent=self.instance_id,
                        agent_type=self.name,
                        tool=display_name,
                        tool_call_id=tool_call_id,
                        status=EventStatus.FAILED,
                        parent_agent=self._parent_agent,
                        wave_id=self._wave_id,
                        details={"error": str(e), **details},
                    )
                raise

        return wrapper

    # --- History Management ---

    def clear_history(self) -> None:
        """Clear conversation history (in-memory and persistent)."""
        self._history = []
        if self._session_id and self._conversation_store:
            self._conversation_store.clear(self._session_id, self.name)
        if self._cache_pipeline:
            self._cache_pipeline.invalidate()

    def _invalidate_cache(self) -> None:
        """Invalidate the context cache.

        Call after changing system_prompt or tools at runtime.
        """
        if self._cache_pipeline:
            self._cache_pipeline.invalidate()

    def close(self) -> None:
        """Clean up agent resources (deletes remote caches, shuts down executor)."""
        if self._cache_pipeline:
            self._cache_pipeline.cleanup()
        if self._cache_executor:
            self._cache_executor.shutdown(wait=False)
            self._cache_executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _save_history(self) -> None:
        """Save conversation history to persistent storage."""
        if self._session_id and self._conversation_store:
            self._conversation_store.save(self._session_id, self.name, self._history)

    # --- Result Summarization ---

    def _summarize_result(self, result: Any, max_length: int = 200) -> str:
        """Create a short summary of tool result for UI display."""
        if result is None:
            return "None"

        if isinstance(result, dict):
            if result.get("error"):
                error_msg = str(result["error"])[:100]
                return f"Error: {error_msg}"

            if "success" in result and "defined_variables" in result:
                if result.get("success"):
                    vars_defined = result.get("defined_variables", [])
                    stdout = result.get("stdout", "").strip()
                    if stdout:
                        first_line = stdout.split("\n")[0][:80]
                        return f"Success: {first_line}"
                    elif vars_defined:
                        return f"Success: defined {', '.join(vars_defined[:5])}"
                    return "Success"
                else:
                    return f"Failed: {result.get('error_type', 'Unknown error')}"

            if "text" in result:
                text = str(result["text"])
                return text[:max_length] + "..." if len(text) > max_length else text

            if "stdout" in result and result["stdout"]:
                stdout = str(result["stdout"])
                return stdout[:max_length] + "..." if len(stdout) > max_length else stdout

            if "output" in result:
                output = str(result["output"])
                return output[:max_length] + "..." if len(output) > max_length else output

            if "result" in result:
                res = str(result["result"])
                return res[:max_length] + "..." if len(res) > max_length else res

            keys = list(result.keys())[:5]
            extra = f" (+{len(result) - 5} more)" if len(result) > 5 else ""
            return f"Dict: {{{', '.join(keys)}{extra}}}"

        if isinstance(result, str):
            return result[:max_length] + "..." if len(result) > max_length else result

        if isinstance(result, (list, tuple)):
            return f"List[{len(result)} items]"

        if isinstance(result, bool):
            return str(result)

        if isinstance(result, (int, float)):
            return str(result)

        result_str = str(result)
        return result_str[:max_length] + "..." if len(result_str) > max_length else result_str

    # --- Context Information ---

    def get_context(self) -> dict:
        """Get the full context window that the agent sees.

        Returns a dict with system prompt and conversation history,
        useful for debugging and visibility into agent state.
        """
        history_formatted = []
        for content in self._history:
            role = content.role
            parts_text = []
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    parts_text.append(part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    parts_text.append(f"[Tool Call: {fc.name}({dict(fc.args) if fc.args else {}})]")
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    parts_text.append(f"[Tool Response: {fr.name} -> {fr.response}]")

            if parts_text:
                history_formatted.append(
                    {
                        "role": role,
                        "content": "\n".join(parts_text),
                    }
                )

        tool_names = list(self._tools.keys())
        context_tokens = self._count_context_tokens()

        return {
            "system_prompt": self.system_prompt,
            "history": history_formatted,
            "tools": tool_names,
            "agent_type": self.name,
            "instance_id": self.instance_id,
            "context_tokens": context_tokens,
        }

    def _count_context_tokens(self) -> int:
        """Count tokens in the current context window."""
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=self.system_prompt)],
                )
            ]
            contents.extend(self._history)

            response = self.client.models.count_tokens(
                model=self.model_name,
                contents=contents,
            )
            return response.total_tokens
        except Exception:
            return 0

    # --- Tool Execution ---

    def _execute_tool(self, function_call: types.FunctionCall) -> tuple[str, Any]:
        """Execute a single tool and return (name, result)."""
        func_name = function_call.name
        args = dict(function_call.args) if function_call.args else {}

        tool_func = self._tools.get(func_name)
        if not tool_func:
            return func_name, {"error": f"Unknown function: {func_name}"}

        try:
            result = tool_func(**args)
            return func_name, result
        except Exception as e:
            return func_name, {"error": str(e)}

    def _execute_tools_parallel(
        self, function_calls: list[types.FunctionCall]
    ) -> list[tuple[str, Any]]:
        """Execute multiple tool calls in parallel."""
        if len(function_calls) == 1:
            return [self._execute_tool(function_calls[0])]

        results = [None] * len(function_calls)
        with ThreadPoolExecutor(
            max_workers=min(len(function_calls), self.MAX_PARALLEL_TOOLS)
        ) as executor:
            future_to_idx = {
                executor.submit(self._execute_tool, fc): idx
                for idx, fc in enumerate(function_calls)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    func_name = function_calls[idx].name
                    results[idx] = (func_name, {"error": str(e)})

        return results

    def _build_function_response_content(self, results: list[tuple[str, Any]]) -> types.Content:
        """Build a Content object with function response parts.

        Supports multimodal responses - if a tool result contains a "files"
        key with file attachments, they are included as Parts in the response.

        Two attachment modes are supported:

        1. **Inline bytes** - Include raw binary data directly:
            {"data": bytes, "mime_type": "application/pdf", "description": "..."}

        2. **GCS URI** - Reference a file in Google Cloud Storage:
            {"gcs_uri": "gs://bucket/path.pdf", "mime_type": "application/pdf", "description": "..."}

        GCS mode is preferred for large files (>10MB) to avoid context limits.

        Expected format:
            {
                "result": "...",
                "files": [
                    {"data": bytes, "mime_type": "image/png", "description": "Chart"},
                    {"gcs_uri": "gs://bucket/paper.pdf", "mime_type": "application/pdf", "description": "Paper"},
                ]
            }

        Note: The legacy "images" key is still supported for backward compatibility.
        """
        parts = []
        for func_name, result in results:
            file_attachments = []
            if isinstance(result, dict):
                # Support both "files" (preferred) and "images" (legacy) keys
                file_attachments = result.pop("files", []) or []
                file_attachments += result.pop("images", []) or []
                response_data = result
            else:
                response_data = {"result": str(result)}

            parts.append(
                types.Part.from_function_response(
                    name=func_name,
                    response=response_data,
                )
            )

            for item in file_attachments:
                if not isinstance(item, dict):
                    continue

                mime_type = item.get("mime_type", "application/octet-stream")
                description = item.get("description", "")

                # Add description as text part if provided
                if description:
                    label = "PDF" if mime_type == "application/pdf" else "File"
                    parts.append(
                        types.Part.from_text(text=f"[{label} from {func_name}: {description}]")
                    )

                # Check for GCS URI first (preferred for large files)
                if "gcs_uri" in item:
                    gcs_uri = item["gcs_uri"]
                    parts.append(
                        types.Part.from_uri(
                            file_uri=gcs_uri,
                            mime_type=mime_type,
                        )
                    )
                    logger.debug(f"Attached GCS file to context: {gcs_uri} ({mime_type})")

                # Fall back to inline bytes
                elif "data" in item:
                    item_data = item["data"]
                    parts.append(
                        types.Part.from_bytes(
                            data=item_data,
                            mime_type=mime_type,
                        )
                    )
                    logger.debug(
                        f"Attached inline {mime_type} to context: "
                        f"{description or func_name} ({len(item_data):,} bytes)"
                    )

        return types.Content(role="user", parts=parts)

    # --- Main Execution Loop ---

    def _run_with_function_loop(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        contents_offset: int = 0,
    ) -> tuple[str, dict]:
        """Run the model with manual function calling loop.

        Args:
            contents: Full conversation history (appended to during loop).
            config: Generation config (may include cached_content).
            contents_offset: Number of leading items covered by cache.
                Only contents[contents_offset:] is sent to generate_content.
        """
        iteration = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        last_prompt_token_count = 0

        while iteration < self.MAX_ITERATIONS:
            iteration += 1

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents[contents_offset:],
                config=config,
            )

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                last_prompt_token_count = getattr(meta, "prompt_token_count", 0) or 0
                total_prompt_tokens += last_prompt_token_count
                total_completion_tokens += getattr(meta, "candidates_token_count", 0) or 0
                cached = getattr(meta, "cached_content_token_count", 0) or 0
                total_cached_tokens += cached
                cache_label = "explicit" if contents_offset > 0 else "implicit"
                pct = cached * 100 // max(last_prompt_token_count, 1) if cached else 0
                logger.info(
                    "[%s] iter=%d cache=%s: %d cached / %d prompt tokens (%d%%)",
                    self.name, iteration, cache_label,
                    cached, last_prompt_token_count, pct,
                )

            token_usage = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "cached_tokens": total_cached_tokens,
                "cache_type": (
                    ("explicit" if contents_offset > 0 else "implicit")
                    if total_cached_tokens else None
                ),
                "last_prompt_token_count": last_prompt_token_count,
                "model": self.model_name,
            }

            if not response.candidates or not response.candidates[0].content:
                return response.text or "", token_usage

            model_content = response.candidates[0].content

            text_parts = []
            function_calls = []
            for part in model_content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    function_calls.append(part.function_call)
                elif hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

            # Emit intermediate text if present alongside tool calls
            if text_parts and function_calls:
                combined_text = "\n".join(text_parts)

                # Call hook
                self.on_model_thinking(combined_text)

                # Emit event if enabled
                if self.emit_lifecycle_events:
                    emit_event(
                        EventType.MODEL_THINKING,
                        agent=self.instance_id,
                        agent_type=self.name,
                        parent_agent=self._parent_agent,
                        wave_id=self._wave_id,
                        details={"text": combined_text},
                    )

            if not function_calls:
                contents.append(model_content)
                self._save_history()
                return response.text or "", token_usage

            contents.append(model_content)
            self._save_history()

            results = self._execute_tools_parallel(function_calls)

            function_response_content = self._build_function_response_content(results)
            contents.append(function_response_content)
            self._save_history()

            # Mid-loop: promote pending cache for cheaper subsequent iterations
            pipeline = self._cache_pipeline
            if pipeline and pipeline.promote_pending():
                if pipeline.is_config_valid(self.system_prompt, self._tool_functions):
                    contents_offset = pipeline.cached_through_index
                    config = types.GenerateContentConfig(
                        cached_content=pipeline.ready_cache_name,
                        temperature=config.temperature,
                        max_output_tokens=config.max_output_tokens,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    )
                    logger.debug(f"Mid-loop cache switch: offset={contents_offset}")
                else:
                    pipeline.invalidate()

            # Emit context update
            if self.emit_lifecycle_events:
                context_tokens = self._count_context_tokens()
                emit_event(
                    EventType.CONTEXT_UPDATE,
                    agent=self.instance_id,
                    agent_type=self.name,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details={"context_tokens": context_tokens},
                )

        token_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "cached_tokens": total_cached_tokens,
            "model": self.model_name,
            "cache_type": (
                ("explicit" if contents_offset > 0 else "implicit")
                if total_cached_tokens else None
            ),
            "last_prompt_token_count": last_prompt_token_count,
        }
        return f"[Max iterations ({self.MAX_ITERATIONS}) reached]", token_usage

    def run(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 32768,
    ) -> str:
        """Run the agent with a prompt (maintains conversation history).

        Args:
            prompt: User prompt or task description.
            temperature: Sampling temperature.
            max_output_tokens: Maximum tokens in response.

        Returns:
            The agent's final text response.
        """
        # Call hook
        self.on_agent_start(prompt)

        # Emit event if enabled
        if self.emit_lifecycle_events:
            emit_event(
                EventType.AGENT_START,
                agent=self.instance_id,
                agent_type=self.name,
                parent_agent=self._parent_agent,
                wave_id=self._wave_id,
                details={"prompt": prompt, "model": self.model_name},
            )

        try:
            base_config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tools=self._tool_functions if self._tool_functions else None,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
            self._history.append(user_content)
            self._save_history()

            # Validate cache config hasn't drifted (system_prompt or tools changed)
            pipeline = self._cache_pipeline
            if pipeline and pipeline.has_ready_cache:
                if not pipeline.is_config_valid(self.system_prompt, self._tool_functions):
                    logger.warning("Cache invalidated: system_prompt or tools changed")
                    pipeline.invalidate()

            if pipeline and pipeline.has_ready_cache:
                cached_config = types.GenerateContentConfig(
                    cached_content=pipeline.ready_cache_name,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                )
                try:
                    result, token_usage = self._run_with_function_loop(
                        self._history,
                        cached_config,
                        contents_offset=pipeline.cached_through_index,
                    )
                except Exception as cache_err:
                    # Cache expired or invalid — fall back to uncached
                    logger.warning(
                        f"Cached generate_content failed, falling back: {cache_err}"
                    )
                    pipeline.invalidate()
                    result, token_usage = self._run_with_function_loop(
                        self._history, base_config
                    )
            else:
                result, token_usage = self._run_with_function_loop(
                    self._history, base_config
                )

            # Post-round cache pipeline: promote pending, fire new
            if pipeline and self._cache_executor:
                pipeline.promote_pending()
                if pipeline.should_cache(
                    self._history,
                    last_prompt_token_count=token_usage.get("last_prompt_token_count"),
                ):
                    pipeline.create_cache_async(
                        contents=list(self._history),
                        system_instruction=self.system_prompt,
                        tools=self._tool_functions if self._tool_functions else None,
                        executor=self._cache_executor,
                    )

            # Call hook
            self.on_agent_end(result, success=True)

            # Emit event if enabled
            if self.emit_lifecycle_events:
                emit_event(
                    EventType.AGENT_END,
                    agent=self.instance_id,
                    agent_type=self.name,
                    status=EventStatus.COMPLETED,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details={"result": result, "token_usage": token_usage},
                )

            return result

        except Exception as e:
            # Call hook
            self.on_agent_end("", success=False, error=str(e))

            # Emit event if enabled
            if self.emit_lifecycle_events:
                emit_event(
                    EventType.AGENT_END,
                    agent=self.instance_id,
                    agent_type=self.name,
                    status=EventStatus.FAILED,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details={"error": str(e)},
                )
            raise

    def run_stateless(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 32768,
    ) -> str:
        """Run the agent without maintaining conversation history.

        Useful for one-shot tasks where each call is independent.

        Args:
            prompt: User prompt or task description.
            context: Optional context dict to include in the prompt.
            temperature: Sampling temperature.
            max_output_tokens: Maximum tokens in response.

        Returns:
            The agent's final text response.
        """
        full_prompt = prompt
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            full_prompt = f"Context:\n{context_str}\n\nTask: {prompt}"

        # Call hook
        self.on_agent_start(full_prompt)

        # Emit event if enabled
        if self.emit_lifecycle_events:
            emit_event(
                EventType.AGENT_START,
                agent=self.instance_id,
                agent_type=self.name,
                parent_agent=self._parent_agent,
                wave_id=self._wave_id,
                details={"prompt": full_prompt, "model": self.model_name},
            )

        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tools=self._tool_functions if self._tool_functions else None,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_prompt)],
                )
            ]

            result, token_usage = self._run_with_function_loop(contents, config)

            # Call hook
            self.on_agent_end(result, success=True)

            # Emit event if enabled
            if self.emit_lifecycle_events:
                emit_event(
                    EventType.AGENT_END,
                    agent=self.instance_id,
                    agent_type=self.name,
                    status=EventStatus.COMPLETED,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details={"result": result, "token_usage": token_usage},
                )

            return result

        except Exception as e:
            # Call hook
            self.on_agent_end("", success=False, error=str(e))

            # Emit event if enabled
            if self.emit_lifecycle_events:
                emit_event(
                    EventType.AGENT_END,
                    agent=self.instance_id,
                    agent_type=self.name,
                    status=EventStatus.FAILED,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details={"error": str(e)},
                )
            raise


def agent_as_tool(agent: Agent, description: str | None = None) -> Callable:
    """Wrap an agent as a tool function for another agent.

    This allows a parent agent to delegate tasks to sub-agents.

    Args:
        agent: The agent to wrap.
        description: Optional description override.

    Returns:
        A callable that can be registered as a tool.
    """

    def tool_func(task: str, context: str = "") -> str:
        """Delegate a task to a specialized agent.

        Args:
            task: The task description for the agent.
            context: Additional context as a string (e.g., JSON).

        Returns:
            The agent's response.
        """
        ctx = None
        if context:
            try:
                import json

                ctx = json.loads(context)
            except json.JSONDecodeError:
                ctx = {"raw_context": context}

        return agent.run_stateless(task, context=ctx)

    tool_func.__name__ = f"{agent.name}_agent"
    tool_func.__doc__ = (
        description
        or f"Delegate task to the {agent.name} specialist agent. {agent.system_prompt[:200]}"
    )

    return tool_func
