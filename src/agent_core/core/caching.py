"""Vertex AI context cache management — Python facade over the C++ manager.

The state machine (slot table, fingerprinting, TTL, pending-future tracking,
background workers, reaper) lives in C++ via
``agent_core._native.cache_manager``. This module keeps a backwards-compatible
``ContextCacheRegistry`` class so existing call sites — including Papyrus —
keep working unchanged. The Python side is responsible only for translating
between provider callables / content lists and the opaque payload that the
create callback understands.

Phase 4 migration notes:
    * ``ContextCacheRegistry`` is now a thin wrapper; instances no longer hold
      slot dictionaries directly. Tests that inspected internal state should
      switch to ``registry.peek_slot(agent_id)``.
    * ``_compute_fingerprint`` continues to compute the same 16-hex-char hash,
      now delegating to the C++ implementation so Python and C++ never disagree.
    * The dataclasses ``CacheAdvice`` and ``_CacheSlot`` are kept as Python
      types for callers that import them by name; the runtime state of an
      agent's slot is, however, mastered in C++.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any, Callable

from google import genai
from google.genai import types

from agent_core import _native

logger = logging.getLogger(__name__)

# Public constant retained from the previous Python implementation.
MIN_TOKEN_GROWTH = 4096


@dataclasses.dataclass(frozen=True, slots=True)
class CacheAdvice:
    """Returned by ``get_advice`` to tell the caller what config to use."""

    cache_name: str | None
    contents_offset: int


@dataclasses.dataclass
class _CacheSlot:
    """Backwards-compatible dataclass.

    The real state lives in C++; this dataclass is only used by code that
    held references to it before the migration. Its fields mirror those in
    the native ``CacheSlotSnapshot``.
    """

    model_name: str
    min_token_threshold: int
    ready_name: str | None = None
    ready_offset: int = 0
    ready_created_at: float | None = None
    pending: Any = None
    pending_through_index: int | None = None
    last_cache_token_count: int = 0
    config_fingerprint: str | None = None


def _compute_fingerprint(
    system_instruction: str | None, tools: list[Callable] | None
) -> str:
    """Compute the cache-config fingerprint exactly as the C++ side does."""
    tool_names = (
        [getattr(f, "__name__", str(f)) for f in tools] if tools else []
    )
    return _native.compute_cache_fingerprint(
        system_instruction or "", tool_names
    )


class ContextCacheRegistry:
    """Process-wide Vertex AI cache registry — facade over the C++ manager.

    Multiple instances can be constructed, but they all share the same
    underlying global state (the C++ ``CacheManager`` is a singleton). The
    last-configured ``client`` / ``ttl`` wins, which matches the previous
    behavior where applications constructed exactly one registry at startup.
    """

    _configured_lock = threading.Lock()

    def __init__(
        self,
        client: genai.Client,
        max_workers: int = 4,
        cache_ttl_seconds: int = 600,
    ) -> None:
        self._client = client
        self._cache_ttl = cache_ttl_seconds
        # Capture model names per agent so we can construct the right Vertex
        # config when the create callback fires.
        self._tool_callables_by_agent: dict[str, list[Callable]] = {}
        self._payload_lock = threading.Lock()
        self._closed = False

        with ContextCacheRegistry._configured_lock:
            _native.cache_manager.configure(
                self._create_remote_cache,
                self._delete_remote_cache,
                max_workers,
                cache_ttl_seconds,
            )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def register(
        self,
        agent_id: str,
        model_name: str,
        min_token_threshold: int = 32_768,
    ) -> None:
        _native.cache_manager.register_agent(
            agent_id, model_name, min_token_threshold
        )

    def unregister(self, agent_id: str) -> None:
        _native.cache_manager.unregister_agent(agent_id)
        with self._payload_lock:
            self._tool_callables_by_agent.pop(agent_id, None)

    def get_advice(
        self,
        agent_id: str,
        system_instruction: str | None,
        tools: list[Callable] | None,
        wait: bool = False,
        wait_timeout: float = 30.0,
    ) -> CacheAdvice:
        fp = _compute_fingerprint(system_instruction, tools)
        native_advice = _native.cache_manager.get_advice(
            agent_id, fp, wait, wait_timeout
        )
        cache_name = native_advice.cache_name or None
        return CacheAdvice(
            cache_name=cache_name,
            contents_offset=native_advice.contents_offset if cache_name else 0,
        )

    def notify(
        self,
        agent_id: str,
        contents: list[types.Content],
        system_instruction: str | None,
        tools: list[Callable] | None,
        token_count: int | None = None,
    ) -> None:
        if self._closed:
            return
        # Resolve the token count via the client API if the caller didn't
        # already supply one — same fallback the old code had.
        snapshot = self._peek_slot(agent_id)
        if snapshot is None:
            # Unregistered agent — match the old code's no-op behavior.
            return
        if token_count is None:
            try:
                resp = self._client.models.count_tokens(
                    model=snapshot.model_name, contents=contents
                )
                token_count = resp.total_tokens
            except Exception as exc:
                logger.debug("Token counting failed, skipping cache: %s", exc)
                return

        fp = _compute_fingerprint(system_instruction, tools)

        # Stash the tool callables so the worker can rebuild the Vertex config
        # later — the C++ payload only carries the contents list and system
        # prompt by value.
        with self._payload_lock:
            self._tool_callables_by_agent[agent_id] = list(tools) if tools else []

        payload = {
            "agent_id": agent_id,
            "model": self._lookup_model_name(agent_id),
            "contents": list(contents),
            "system_instruction": system_instruction,
            "ttl": f"{self._cache_ttl + 60}s",
        }
        _native.cache_manager.notify(
            agent_id, fp, int(token_count), len(contents), payload
        )

    def invalidate(self, agent_id: str) -> None:
        _native.cache_manager.invalidate(agent_id)

    def close(self) -> None:
        """Tear down this facade.

        The C++ cache manager is process-global; we don't stop its workers
        here (the module's atexit handler does). We just drop our reference
        to per-agent payload state and mark this facade as closed.
        """
        self._closed = True
        with self._payload_lock:
            self._tool_callables_by_agent.clear()

    # ------------------------------------------------------------------
    # Introspection helpers (mostly for tests)
    # ------------------------------------------------------------------

    def peek_slot(self, agent_id: str):
        """Return a snapshot of the slot, or ``None`` if not registered.

        The returned object is a ``_native.CacheSlotSnapshot`` — a frozen view
        with these read-only attributes:

        * ``model_name`` / ``min_token_threshold``
        * ``has_ready`` / ``ready_name`` / ``ready_offset`` / ``ready_expired``
        * ``pending`` (bool) / ``pending_through_index``
        * ``last_cache_token_count`` / ``config_fingerprint``
        """
        return _native.cache_manager.peek_slot(agent_id)

    def seed_slot(
        self,
        agent_id: str,
        *,
        ready_name: str = "",
        ready_offset: int = 0,
        ready_expired: bool = False,
        pending_done: bool = False,
        pending_cache_name: str = "",
        pending_through_index: int = 0,
        last_cache_token_count: int = 0,
        config_fingerprint: str = "",
    ) -> None:
        """Force a slot into a particular state — test helper."""
        _native.cache_manager.seed_slot(
            agent_id,
            ready_name,
            ready_offset,
            ready_expired,
            pending_done,
            pending_cache_name,
            pending_through_index,
            last_cache_token_count,
            config_fingerprint,
        )

    def _peek_slot(self, agent_id: str) -> Any:
        return _native.cache_manager.peek_slot(agent_id)

    def _lookup_model_name(self, agent_id: str) -> str:
        snap = _native.cache_manager.peek_slot(agent_id)
        if snap is None:
            raise RuntimeError(
                f"cache notify for unregistered agent {agent_id}"
            )
        return snap.model_name

    # ------------------------------------------------------------------
    # Callbacks invoked from C++ workers
    # ------------------------------------------------------------------

    def _create_remote_cache(self, payload: dict) -> str:
        """Worker callback — turns the opaque payload into a Vertex API call."""
        try:
            agent_id = payload["agent_id"]
            with self._payload_lock:
                tools = self._tool_callables_by_agent.get(agent_id, [])
            converted_tools = self._convert_tools(tools) if tools else None
            cache = self._client.caches.create(
                model=payload["model"],
                config=types.CreateCachedContentConfig(
                    contents=payload["contents"],
                    system_instruction=payload["system_instruction"],
                    tools=converted_tools,
                    ttl=payload["ttl"],
                ),
            )
            logger.debug("Cache created for %s: %s", agent_id, cache.name)
            return cache.name or ""
        except Exception as exc:
            logger.warning("Cache creation failed: %s", exc)
            return ""

    def _delete_remote_cache(self, cache_name: str) -> None:
        try:
            self._client.caches.delete(name=cache_name)
            logger.debug("Cache deleted: %s", cache_name)
        except Exception as exc:
            logger.debug("Cache deletion failed (API TTL will handle): %s", exc)

    def _convert_tools(self, tools: list[Callable]) -> list[types.Tool]:
        declarations = [
            types.FunctionDeclaration.from_callable(
                callable=f, client=self._client._api_client
            )
            for f in tools
        ]
        return [types.Tool(function_declarations=declarations)]
