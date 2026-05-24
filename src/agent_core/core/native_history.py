"""Python wrapper around the native ``HistoryStore``.

The agent loop treats history as a list of opaque provider messages
(``contents.append(...)``, ``contents[:] = [...]``, ``len(contents)``,
iteration). ``NativeHistory`` is a thin shim that forwards each of those
operations into the C++ store while keeping the wire-format conversion
(provider-native ↔ canonical JSON) on the Python side, where the providers
live.

The class is intentionally minimal — every public method maps to one or two
C++ calls. Subclassing is not supported; the agent loop owns one instance per
session.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterator

from agent_core import _native

if TYPE_CHECKING:
    from agent_core.providers.types import LLMProvider


class NativeHistory:
    """List-shaped view backed by the C++ ``HistoryStore``.

    Construction either opens a fresh session or — if ``db_path`` is supplied
    and a row exists for ``(session_id, agent_type)`` — rehydrates the
    canonical state and re-attaches ``provider_native`` references by calling
    ``provider.from_canonical()`` for each persisted message.
    """

    __slots__ = ("_provider", "_store", "_provider_tag")

    def __init__(
        self,
        provider: "LLMProvider",
        session_id: str = "",
        agent_type: str = "",
        db_path: str = "",
        *,
        store: Any = None,
    ) -> None:
        self._provider = provider
        self._provider_tag = getattr(provider, "PROVIDER_TAG", "")
        if store is not None:
            self._store = store
        else:
            self._store = _native.HistoryStore(session_id, agent_type, db_path)
        self._reattach_natives_after_load()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _reattach_natives_after_load(self) -> None:
        """Rebuild provider_native refs for slots loaded from SQLite."""
        natives = self._store.snapshot_native()
        canonical = self._store.snapshot_canonical()
        roles = [r for r, _ in self._store.snapshot_role_and_json()]

        # If no slot has a None placeholder, nothing to do — this is the
        # in-process construction path, not a reload.
        if not any(n is None for n in natives):
            return

        from agent_core.providers.types import CanonicalMessage

        slots: list[tuple[Any, str, int, str, str]] = []
        for raw_json, role in zip(canonical, roles):
            approx_tokens = max(1, (len(raw_json) + 3) // 4)
            stripped = CanonicalMessage(
                role=role,  # type: ignore[arg-type]
                provider_tag=self._provider_tag,
                canonical_json=raw_json,
                approx_tokens=approx_tokens,
                provider_native=None,
            )
            native = self._provider.from_canonical(stripped)
            slots.append((native, raw_json, approx_tokens, role, self._provider_tag))
        self._store.rehydrate_canonical(slots)

    # ------------------------------------------------------------------
    # List-shaped API consumed by the agent loop
    # ------------------------------------------------------------------

    def append(self, message: Any) -> None:
        c = self._provider.to_canonical(message)
        self._store.append(
            c.provider_native if c.provider_native is not None else message,
            c.canonical_json,
            c.approx_tokens,
            c.role,
            c.provider_tag,
        )

    def extend(self, messages: list[Any]) -> None:
        for m in messages:
            self.append(m)

    def clear(self) -> None:
        self._store.clear()

    def replace_all(self, messages: list[Any]) -> None:
        """Swap the entire history for *messages*. Used by compaction's
        ``contents[:] = [...]`` slice replacement."""
        self._store.clear()
        for m in messages:
            self.append(m)

    def replace_prefix(self, prefix_len: int, summary: Any) -> None:
        """Collapse the first *prefix_len* messages into *summary*."""
        c = self._provider.to_canonical(summary)
        self._store.replace_prefix(
            prefix_len,
            c.provider_native if c.provider_native is not None else summary,
            c.canonical_json,
            c.approx_tokens,
            c.role,
            c.provider_tag,
        )

    def flush(self) -> None:
        """Block until persistence catches up."""
        self._store.flush()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def snapshot(self) -> list[Any]:
        """Return a fresh list of provider-native messages."""
        return self._store.snapshot_native()

    def canonical_snapshot(self) -> list[str]:
        """Return a fresh list of canonical JSON strings."""
        return self._store.snapshot_canonical()

    def total_approx_tokens(self) -> int:
        return self._store.total_approx_tokens()

    # ------------------------------------------------------------------
    # Sequence protocol — these are what the agent loop and tests rely on
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._store.snapshot_native())

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._store.snapshot_native()[index]
        return self._store.get_native(index)

    def __setitem__(self, index, value) -> None:
        # The only slice-write pattern the agent loop uses is ``contents[:] =
        # [...]`` (full replacement during compaction). Reject anything else
        # rather than silently falling out of sync with the C++ store.
        if isinstance(index, slice) and index == slice(None, None, None):
            self.replace_all(list(value))
            return
        raise TypeError(
            "NativeHistory only supports full-slice assignment "
            "(contents[:] = [...])"
        )

    def __bool__(self) -> bool:
        return len(self) > 0

    def __eq__(self, other: object) -> bool:
        # Comparing history to a Python list is how a few existing tests verify
        # state. We compare via the native snapshot so identity-based assertions
        # still work when the test holds references to the original messages.
        if isinstance(other, list):
            return self._store.snapshot_native() == other
        if isinstance(other, NativeHistory):
            return (
                self._store.snapshot_native() == other._store.snapshot_native()
            )
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"NativeHistory(session_id={self._store.session_id!r}, "
            f"agent_type={self._store.agent_type!r}, len={len(self)})"
        )
