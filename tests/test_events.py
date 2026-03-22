"""Tests for the event system."""

import time

import pytest

from agent_core.core.events import (
    EventType,
    EventStatus,
    Event,
    EventBus,
    get_event_bus,
    emit_event,
)


class TestEvent:
    """Test Event dataclass."""

    def test_event_creation(self):
        event = Event(
            type=EventType.AGENT_START,
            agent="designer_abc123",
            agent_type="designer",
        )
        assert event.type == EventType.AGENT_START
        assert event.agent == "designer_abc123"
        assert event.status == EventStatus.RUNNING
        assert event.details == {}
        assert event.timestamp > 0

    def test_event_to_dict(self):
        event = Event(
            type=EventType.TOOL_START,
            agent="coder_xyz",
            agent_type="coder",
            tool="execute_code",
            tool_call_id="exec_123",
            details={"code": "print('hi')"},
        )
        d = event.to_dict()
        assert d["type"] == "tool_start"
        assert d["agent"] == "coder_xyz"
        assert d["tool"] == "execute_code"
        assert d["details"]["code"] == "print('hi')"

    def test_event_with_custom_string_type(self):
        event = Event(type="custom_event", agent="test_agent")
        d = event.to_dict()
        assert d["type"] == "custom_event"


class TestEventBus:
    """Test EventBus pub/sub."""

    def test_emit_and_subscribe(self):
        bus = EventBus()
        received = []

        bus.subscribe(lambda e: received.append(e))
        event = Event(type=EventType.AGENT_START, agent="test")
        bus.emit(event)

        assert len(received) == 1
        assert received[0] is event

    def test_multiple_subscribers(self):
        bus = EventBus()
        received_a, received_b = [], []

        bus.subscribe(lambda e: received_a.append(e))
        bus.subscribe(lambda e: received_b.append(e))
        bus.emit(Event(type=EventType.AGENT_END, agent="test"))

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_subscriber_error_does_not_block_others(self):
        bus = EventBus()
        received = []

        def bad_subscriber(e):
            raise ValueError("boom")

        bus.subscribe(bad_subscriber)
        bus.subscribe(lambda e: received.append(e))
        bus.emit(Event(type=EventType.ERROR, agent="test"))

        # Second subscriber should still receive the event
        assert len(received) == 1
        # Failure should be tracked
        assert len(bus.get_failed_deliveries()) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []

        callback = lambda e: received.append(e)
        bus.subscribe(callback)
        bus.emit(Event(type=EventType.AGENT_START, agent="test"))
        assert len(received) == 1

        bus.unsubscribe(callback)
        bus.emit(Event(type=EventType.AGENT_START, agent="test"))
        assert len(received) == 1  # no new events

    def test_get_events(self):
        bus = EventBus()
        bus.emit(Event(type=EventType.AGENT_START, agent="a"))
        bus.emit(Event(type=EventType.AGENT_END, agent="a"))

        events = bus.get_events()
        assert len(events) == 2
        # Should be a copy
        events.append(Event(type=EventType.ERROR, agent="x"))
        assert len(bus.get_events()) == 2

    def test_clear(self):
        bus = EventBus()
        bus.emit(Event(type=EventType.AGENT_START, agent="a"))
        bus.clear()
        assert len(bus.get_events()) == 0

    def test_get_recent(self):
        bus = EventBus()
        e1 = Event(type=EventType.AGENT_START, agent="a")
        bus.emit(e1)
        cutoff = time.time()
        time.sleep(0.01)
        e2 = Event(type=EventType.AGENT_END, agent="a")
        bus.emit(e2)

        recent = bus.get_recent(since_timestamp=cutoff)
        assert len(recent) == 1
        assert recent[0] is e2

    def test_health_status(self):
        bus = EventBus()
        status = bus.get_health_status()
        assert status["subscriber_count"] == 0
        assert status["total_events"] == 0
        assert status["is_healthy"] is True


class TestEmitEventHelper:
    """Test the module-level emit_event convenience function."""

    def test_emit_event_returns_event(self):
        # Reset global bus
        import agent_core.core.events as events_module
        events_module._event_bus = None

        event = emit_event(
            EventType.TOOL_START,
            agent="test_agent",
            agent_type="tester",
            tool="my_tool",
        )

        assert isinstance(event, Event)
        assert event.type == EventType.TOOL_START
        assert event.agent == "test_agent"
        assert event.tool == "my_tool"

        # Should be in the global bus
        bus = get_event_bus()
        assert event in bus.get_events()

        # Clean up
        events_module._event_bus = None
