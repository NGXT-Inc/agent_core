# Agent Core

Extensible agent orchestration framework with multi-provider function calling support.

## Installation

### From another project (editable install)

```bash
pip install -e /path/to/LDIA/agent_core
```

Or in your `requirements.txt`:
```
-e /path/to/LDIA/agent_core
```

### Within LDIA

Already configured as a local dependency in LDIA's `pyproject.toml`.

## Quick Start

```python
from agent_core import Agent, EventType, emit_event

class ResearcherAgent(Agent):
    name = "researcher"
    system_prompt = "You are a literature research assistant..."

    # Optional: disable default events if you handle them yourself
    # emit_lifecycle_events = False
    # emit_tool_events = False

    def __init__(self, session_id: str = None):
        super().__init__(session_id=session_id)
        self.register_tool(self.search_papers)
        self.register_tool(self.read_paper)

    def search_papers(self, query: str, max_results: int = 10) -> list[dict]:
        """Search for research papers matching the query."""
        # Your implementation
        return [{"title": "...", "abstract": "..."}]

    def read_paper(self, paper_id: str) -> dict:
        """Read the full content of a paper."""
        # Your implementation
        return {"content": "..."}

    # Optional: override hooks for custom behavior
    def on_tool_start(self, tool_name: str, args: dict, tool_call_id: str):
        print(f"Starting {tool_name}...")

    def on_tool_end(self, tool_name: str, args: dict, tool_call_id: str,
                    result, success: bool, error: str = None):
        if success:
            print(f"Completed {tool_name}")
        else:
            print(f"Failed {tool_name}: {error}")

# Use the agent
agent = ResearcherAgent()
response = agent.run("Find papers about transformer architectures")
```

### Streaming Text

User-facing callers can opt in to incremental text deltas while preserving the
normal final return value and conversation history:

```python
response = agent.run(
    "Explain recent work on retrieval-augmented generation",
    streaming=True,
    on_text_delta=lambda delta: print(delta, end="", flush=True),
)
```

You can also make streaming the default for an agent and override it per call:

```python
class StreamingResearcherAgent(ResearcherAgent):
    DEFAULT_STREAMING = True

agent = StreamingResearcherAgent()
response = agent.run("Explain recent work on retrieval-augmented generation")
plain_response = agent.run("Summarize this briefly", streaming=False)
```

Streaming is transport-agnostic: applications decide whether deltas go to SSE,
websockets, a terminal, or nowhere. The final response is still returned from
`run()` and persisted through the configured conversation store.

### File Attachments And Rich Responses

Attach files to user messages with provider-neutral `FilePart` objects:

```python
from agent_core import FilePart

response = agent.run(
    "Summarize this document",
    attachments=[
        FilePart.from_path("paper.pdf", mime_type="application/pdf"),
    ],
)
```

`run()` still returns the final text for backward compatibility. Use
`run_response()` when a provider may return richer output, such as generated
images:

```python
response = agent.run_response("Generate a diagram of the workflow")
print(response.text)
for part in response.parts:
    print(type(part), getattr(part, "mime_type", None))
```

Attachments are ephemeral in built-in persistence. Live requests include the
file bytes, URI, or provider file ID, but saved/reloaded histories retain only a
text placeholder with the filename/type so the model can see that an attachment
was present. Applications that need durable file context should store files in
their own attachment store and reattach them on follow-up messages.

### OpenRouter Models And Caching

Use `OpenRouterProvider` for OpenRouter-hosted models such as Kimi and
DeepSeek:

```python
from agent_core import Agent, OpenRouterProvider

provider = OpenRouterProvider(
    response_cache=True,
    response_cache_ttl_seconds=300,
)
agent = Agent(
    provider=provider,
    model_name="moonshotai/kimi-k2.6",
    streaming=True,
)
```

OpenRouter provider-side prompt caching is automatic for supported providers
such as Moonshot/Kimi and DeepSeek. `OpenRouterProvider` also supports
OpenRouter response caching for identical requests, preserves reasoning traces,
streams responses, and parses cache usage from
`prompt_tokens_details.cached_tokens` and `cache_write_tokens`.

Gemini's explicit Vertex context cache remains available through
`ContextCacheRegistry`. That cache registry is only used by Gemini providers;
OpenRouter and other OpenAI-compatible providers use request-level cache
controls instead.

## Configuration

### Class Attributes

Override these in your agent subclass:

| Attribute | Default | Description |
|-----------|---------|-------------|
| `name` | `"base"` | Agent type identifier |
| `system_prompt` | `"You are a helpful assistant."` | System prompt |
| `DEFAULT_MODEL` | `"gemini-3.1-pro-preview"` | Model to use |
| `ROOT_AGENT_TYPES` | `{"designer", "analyst", "data_analyst"}` | Types that get deterministic IDs |
| `CODE_TOOLS` | `{"execute_code", ...}` | Tools that handle code (special formatting) |
| `MAX_PARALLEL_TOOLS` | `10` | Max concurrent tool executions |
| `MAX_ITERATIONS` | `50` | Max function calling loop iterations |
| `DEFAULT_STREAMING` | `False` | Use provider streaming by default |
| `emit_lifecycle_events` | `True` | Emit AGENT_START/END events |
| `emit_tool_events` | `True` | Emit TOOL_START/END events |

### Lifecycle Hooks

Override these methods for custom behavior:

```python
def on_agent_start(self, prompt: str) -> None:
    """Called when agent starts processing."""

def on_agent_end(self, result: str, success: bool, error: str = None) -> None:
    """Called when agent finishes processing."""

def on_tool_start(self, tool_name: str, args: dict, tool_call_id: str) -> None:
    """Called before a tool executes."""

def on_tool_end(self, tool_name: str, args: dict, tool_call_id: str,
                result, success: bool, error: str = None) -> None:
    """Called after a tool finishes."""

def on_model_thinking(self, text: str) -> None:
    """Called when model emits intermediate reasoning."""
```

## Persistence

### No persistence (default without session_id)

```python
agent = MyAgent()  # In-memory history only
```

### SQLite persistence

```python
from agent_core.core.persistence import SQLiteConversationStore

store = SQLiteConversationStore("~/.myapp/conversations.db")
agent = MyAgent(session_id="session123", conversation_store=store)
```

### Custom persistence

Implement the `ConversationStoreProtocol`:

```python
from agent_core.core.persistence import ConversationStoreProtocol

class RedisConversationStore:
    def load(self, session_id: str, agent_type: str) -> list:
        # Load from Redis
        ...

    def save(self, session_id: str, agent_type: str, history: list) -> None:
        # Save to Redis
        ...

    def clear(self, session_id: str, agent_type: str) -> None:
        # Clear from Redis
        ...

store = RedisConversationStore()
agent = MyAgent(session_id="session123", conversation_store=store)
```

## Events

### Core Event Types

| Event | Description |
|-------|-------------|
| `AGENT_START` | Agent begins processing |
| `AGENT_END` | Agent finishes (success or failure) |
| `TOOL_START` | Tool execution begins |
| `TOOL_END` | Tool execution completes |
| `MODEL_THINKING` | Intermediate reasoning text |
| `CONTEXT_UPDATE` | Context window token count changed |
| `ERROR` | Error occurred |

### Subscribing to Events

```python
from agent_core import get_event_bus

def my_handler(event):
    print(f"{event.type}: {event.agent}")

bus = get_event_bus()
bus.subscribe(my_handler)
```

### Custom Event Types

Use string values for custom events:

```python
from agent_core import emit_event

emit_event(
    "paper_downloaded",  # Custom event type as string
    agent=self.instance_id,
    agent_type=self.name,
    details={"paper_id": "arxiv:1234", "size_mb": 5.2}
)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_PROJECT_ID` | For Gemini auto-client creation | Google Cloud project ID |
| `GOOGLE_LOCATION` | No | Vertex AI location (default: "global") |
| `OPENROUTER_API_KEY` | For OpenRouter auto-client creation | OpenRouter API key |
