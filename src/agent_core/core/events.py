"""Event system for tracking agent and tool activity.

Events are emitted as agents and tools execute, allowing the UI
to display real-time progress visualization.

This module contains only core/universal event types. Domain-specific
events (experiments, notebooks, etc.) should be defined in the application.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Core event types for agent orchestration.

    Applications can extend this by defining their own EventType enums
    and using string values in the Event.type field.
    """

    # Agent lifecycle (universal)
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"

    # Tool execution (universal)
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"

    # Model activity (universal)
    MODEL_THINKING = "model_thinking"  # Intermediate text during tool-calling loop
    CONTEXT_UPDATE = "context_update"  # Context window token count update
    CONTEXT_COMPACTION = "context_compaction"  # Conversation history compacted

    # Errors (universal)
    ERROR = "error"


class EventStatus(str, Enum):
    """Status of an event."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Event:
    """Represents an activity event.

    This is a minimal event structure containing only universal fields.
    Domain-specific data should be stored in the `details` dict.

    Examples of domain-specific data in details:
        - Experiment tracking: details={"experiment_id": "exp_123", "metrics": {...}}
        - Notebook events: details={"cell_index": 5, "cell_id": "abc"}
        - Custom tools: details={"custom_field": "value"}
    """

    type: EventType | str  # Allow string for custom event types
    agent: str  # Unique instance ID (e.g., "designer_abc123")
    agent_type: str | None = None  # Agent type (e.g., "designer")
    status: EventStatus = EventStatus.RUNNING
    tool: str | None = None
    tool_call_id: str | None = None  # Unique ID to pair TOOL_START with TOOL_END
    parent_agent: str | None = None  # Parent's instance ID
    wave_id: str | None = None  # Wave identifier for grouping parallel agents
    details: dict = field(default_factory=dict)  # Domain-specific data goes here
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert event to dictionary for serialization."""
        return {
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "agent": self.agent,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "tool": self.tool,
            "tool_call_id": self.tool_call_id,
            "parent_agent": self.parent_agent,
            "wave_id": self.wave_id,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class EventBus:
    """Simple event bus for publishing and subscribing to events.

    Provides robust error handling and delivery tracking for event broadcasting.

    Args:
        max_events: Maximum number of events to retain. Oldest events are
            discarded when the limit is reached. Defaults to 10_000.
            Set to 0 for unlimited (not recommended for long-running apps).
    """

    def __init__(self, max_events: int = 10_000):
        self._events: deque[Event] = deque(maxlen=max_events or None)
        self._max_events = max_events
        self._subscribers: list[Callable[[Event], None]] = []
        # Track delivery failures for debugging
        self._failed_deliveries: deque[dict] = deque(maxlen=100)

    def emit(self, event: Event) -> None:
        """Emit an event to all subscribers.

        Errors in individual subscribers are logged and tracked but don't
        prevent delivery to other subscribers.
        """
        self._events.append(event)

        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception as e:
                self._failed_deliveries.append({
                    "event_type": event.type.value if isinstance(event.type, Enum) else event.type,
                    "agent": event.agent,
                    "error": str(e),
                    "timestamp": event.timestamp,
                })
                logger.warning(f"Subscriber error for {event.type}: {e}")

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """Subscribe to events."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def get_events(self) -> list[Event]:
        """Get all events."""
        return list(self._events)

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()

    def get_recent(self, since_timestamp: float | None = None) -> list[Event]:
        """Get events since a timestamp."""
        if since_timestamp is None:
            return list(self._events)
        return [e for e in self._events if e.timestamp > since_timestamp]

    def get_failed_deliveries(self) -> list[dict]:
        """Get recent failed event deliveries for debugging."""
        return list(self._failed_deliveries)

    def get_health_status(self) -> dict:
        """Get health status of the event bus."""
        return {
            "subscriber_count": len(self._subscribers),
            "total_events": len(self._events),
            "max_events": self._max_events,
            "recent_failures": len(self._failed_deliveries),
            "is_healthy": len(self._failed_deliveries) < 10,
        }


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def emit_event(
    event_type: EventType | str,
    agent: str,
    agent_type: str | None = None,
    status: EventStatus = EventStatus.RUNNING,
    tool: str | None = None,
    tool_call_id: str | None = None,
    parent_agent: str | None = None,
    wave_id: str | None = None,
    details: dict | None = None,
) -> Event:
    """Convenience function to emit an event.

    Args:
        event_type: The type of event (EventType enum or custom string).
        agent: Unique instance ID of the agent.
        agent_type: Type of agent (e.g., "designer", "researcher").
        status: Event status (RUNNING, COMPLETED, FAILED).
        tool: Tool name if this is a tool event.
        tool_call_id: Unique ID to pair TOOL_START with TOOL_END.
        parent_agent: Parent agent's instance ID for hierarchy tracking.
        wave_id: Wave ID for grouping parallel agent spawns.
        details: Domain-specific data dict.

    Returns:
        The emitted Event object.
    """
    event = Event(
        type=event_type,
        agent=agent,
        agent_type=agent_type,
        status=status,
        tool=tool,
        tool_call_id=tool_call_id,
        parent_agent=parent_agent,
        wave_id=wave_id,
        details=details or {},
    )
    get_event_bus().emit(event)
    return event
