"""Process-global session registry — Python read-only proxy.

Every ``Agent`` constructed with a ``session_id`` acquires a slot in the C++
``Registry`` singleton. Other scripts in the same process can introspect or
operate on those sessions through this module without having to import the
underlying ``_native`` extension directly.

The Python surface intentionally exposes only the safe, idempotent operations.
``acquire`` / ``release`` live on the ``Agent`` lifecycle and are not surfaced
here — scripts that want a session for inspection but not control should call
``peek``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_core import _native

if TYPE_CHECKING:
    pass


def list_active() -> list[str]:
    """Return every resident session id in sorted order."""
    return _native.registry.list_active()


def descendants(root_session_id: str, *, include_root: bool = False) -> list[str]:
    """Return resident session ids under *root_session_id*.

    With ``include_root=True`` the root itself is included if it is resident.
    """
    return _native.registry.descendants(root_session_id, include_root)


def cancel_subtree(root_session_id: str) -> int:
    """Flag *root_session_id* and every descendant as cancelled.

    Returns the number of sessions flagged. Cancellation is a *signal*; the
    agent loop checks it at safe poll points and unwinds its current iteration.
    """
    return _native.registry.cancel_subtree(root_session_id)


def clear_cancellation(session_id: str, *, recursive: bool = False) -> None:
    """Clear the cancellation flag for a single session or a subtree.

    Typically not needed — ``Agent.run()`` clears its own session's flag at the
    start of every run. Exposed for cases where an external script wants to
    revive a subtree without spinning up agents one by one.
    """
    _native.registry.clear_cancellation(session_id, recursive)


def peek(session_id: str) -> Any:
    """Return a non-counted handle for *session_id*, or ``None``.

    Holding the returned handle does **not** keep the session alive — it just
    lets you query the cancellation flag, access history, etc. for as long as
    the session is resident.
    """
    return _native.registry.peek(session_id)


def session_info(session_id: str) -> dict | None:
    """Return a small dict describing *session_id*, or ``None`` if not resident."""
    handle = _native.registry.peek(session_id)
    if handle is None:
        return None
    history = handle.history()
    return {
        "session_id": handle.session_id,
        "agent_type": handle.agent_type,
        "db_path": handle.db_path,
        "is_cancelled": handle.is_cancelled,
        "ref_count": handle.ref_count,
        "message_count": len(history),
        "total_approx_tokens": history.total_approx_tokens(),
    }


def set_idle_ttl_seconds(seconds: int) -> None:
    """Configure how long an idle session stays resident before being evicted.

    Defaults to 1800 (30 minutes). Eviction does not drop persisted state —
    the next ``acquire`` of the same session_id resurrects from SQLite.
    """
    _native.registry.set_idle_ttl_seconds(int(seconds))


def idle_ttl_seconds() -> int:
    """Return the current idle TTL in seconds."""
    return _native.registry.idle_ttl_seconds()


def reap_now() -> int:
    """Synchronously evict idle sessions. Returns the number evicted."""
    return _native.registry.reap_now()


__all__ = [
    "list_active",
    "descendants",
    "cancel_subtree",
    "clear_cancellation",
    "peek",
    "session_info",
    "set_idle_ttl_seconds",
    "idle_ttl_seconds",
    "reap_now",
]
