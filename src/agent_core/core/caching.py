"""Centralized Vertex AI context cache management for agent conversations.

Manages per-agent cache slots from a shared executor pool. Each agent
registers a slot, and the registry handles background cache creation,
auto-promotion, TTL enforcement, and guaranteed cleanup of remote caches.

Every generate_content call should query get_advice() for the best
available cache. The registry auto-promotes pending caches on every
query, so callers never need to manage cache state manually.

This module is internal to agent_core. Developers using the package
interact with it indirectly through Agent.init_cache_registry().
"""

import dataclasses
import hashlib
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Minimum new (uncached) tokens before considering a new cache creation.
MIN_TOKEN_GROWTH = 4096


@dataclasses.dataclass(frozen=True, slots=True)
class CacheAdvice:
    """Returned by get_advice() to tell the caller what config to use.

    If cache_name is None, use the normal base config (with system_instruction
    and tools) and send all contents. If cache_name is set, build a config
    with cached_content=cache_name (no system_instruction or tools) and send
    only contents[contents_offset:].
    """

    cache_name: str | None
    contents_offset: int


@dataclasses.dataclass
class _CacheSlot:
    """Per-agent cache state managed by the registry."""

    model_name: str
    min_token_threshold: int

    # Ready cache (currently usable)
    ready_name: str | None = None
    ready_offset: int = 0
    ready_created_at: float | None = None  # time.monotonic()

    # Pending cache (being created in background)
    pending: Future | None = None
    pending_through_index: int | None = None

    # Tracking for cache creation decisions
    last_cache_token_count: int = 0
    config_fingerprint: str | None = None


class ContextCacheRegistry:
    """Singleton registry managing Vertex AI context caches for all agents.

    Provides a shared executor pool and per-agent cache slots with
    automatic promotion, TTL enforcement, and guaranteed cleanup.
    """

    def __init__(
        self,
        client: genai.Client,
        max_workers: int = 4,
        cache_ttl_seconds: int = 600,
    ) -> None:
        self._client = client
        self._cache_ttl = cache_ttl_seconds
        # API-side TTL: our TTL + 60s safety buffer
        self._api_ttl = f"{cache_ttl_seconds + 60}s"

        self._lock = threading.Lock()
        self._slots: dict[str, _CacheSlot] = {}
        self._closed = False

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="cache-registry"
        )

        # Background reaper cleans up expired caches from idle agents
        self._reaper_stop = threading.Event()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop, daemon=True, name="cache-reaper"
        )
        self._reaper_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        agent_id: str,
        model_name: str,
        min_token_threshold: int = 32_768,
    ) -> None:
        """Register an agent slot. Called once during agent init."""
        with self._lock:
            if agent_id in self._slots:
                logger.warning("Agent %s already registered, re-registering", agent_id)
                old_slot = self._slots[agent_id]
                names = self._collect_cache_names(old_slot)
                for name in names:
                    self._delete_cache(name)
            self._slots[agent_id] = _CacheSlot(
                model_name=model_name,
                min_token_threshold=min_token_threshold,
            )
            logger.debug("Registered cache slot for %s", agent_id)

    def unregister(self, agent_id: str) -> None:
        """Remove agent slot and delete its remote caches."""
        names_to_delete: list[str] = []
        with self._lock:
            slot = self._slots.pop(agent_id, None)
            if slot is None:
                return
            names_to_delete = self._collect_cache_names(slot, wait_for_pending=True)

        # Delete outside lock (I/O)
        for name in names_to_delete:
            self._delete_cache(name)
        logger.debug("Unregistered cache slot for %s", agent_id)

    def get_advice(
        self,
        agent_id: str,
        system_instruction: str | None,
        tools: list[Callable] | None,
        wait: bool = False,
        wait_timeout: float = 30.0,
    ) -> CacheAdvice:
        """Return the best available cache for this agent.

        Auto-promotes pending caches and validates config on every call.

        Args:
            agent_id: The agent to get advice for.
            system_instruction: Current system prompt (for config drift check).
            tools: Current tool list (for config drift check).
            wait: If True and a cache creation is pending, block until it
                completes (up to wait_timeout seconds) before returning.
                Useful when the caller wants to guarantee a cache hit.
            wait_timeout: Maximum seconds to wait for a pending cache.
                Only used when wait=True. Defaults to 30s.
        """
        no_cache = CacheAdvice(cache_name=None, contents_offset=0)
        old_name: str | None = None

        # If wait=True, block for the pending future outside the lock
        if wait:
            pending = None
            with self._lock:
                slot = self._slots.get(agent_id)
                if slot is not None and slot.pending is not None and not slot.pending.done():
                    pending = slot.pending
            if pending is not None:
                try:
                    pending.result(timeout=wait_timeout)
                except Exception:
                    pass  # Promotion will handle the failure

        with self._lock:
            slot = self._slots.get(agent_id)
            if slot is None:
                return no_cache

            self._try_promote(slot)

            if slot.ready_name is None:
                return no_cache

            # Invalidate on TTL expiry or config drift
            if self._is_expired(slot):
                old_name = slot.ready_name
                self._clear_ready(slot)
            elif slot.config_fingerprint is not None:
                current_fp = _compute_fingerprint(system_instruction, tools)
                if current_fp != slot.config_fingerprint:
                    logger.warning(
                        "Cache invalidated for %s: config drift", agent_id
                    )
                    old_name = slot.ready_name
                    self._reset_slot_locked(slot)

            if old_name is None:
                return CacheAdvice(
                    cache_name=slot.ready_name,
                    contents_offset=slot.ready_offset,
                )

        # Delete stale cache outside lock
        self._delete_cache(old_name)
        return no_cache

    def notify(
        self,
        agent_id: str,
        contents: list[types.Content],
        system_instruction: str | None,
        tools: list[Callable] | None,
        token_count: int | None = None,
    ) -> None:
        """Notify that an agent's history has changed.

        Attempts promotion, then fires a new cache creation if thresholds
        are met. Called after generate_content rounds and after tool
        results are appended.
        """
        # Resolve token count outside the lock to avoid blocking I/O
        if token_count is None:
            model_name = None
            with self._lock:
                slot = self._slots.get(agent_id)
                if slot is not None:
                    model_name = slot.model_name
            if model_name is None:
                return
            try:
                response = self._client.models.count_tokens(
                    model=model_name, contents=contents
                )
                token_count = response.total_tokens
            except Exception as e:
                logger.debug("Token counting failed, skipping cache: %s", e)
                return

        with self._lock:
            if self._closed:
                return
            slot = self._slots.get(agent_id)
            if slot is None:
                return

            self._try_promote(slot)

            if not self._should_cache(slot, token_count):
                return

            self._fire_creation(
                slot=slot,
                agent_id=agent_id,
                contents=list(contents),
                system_instruction=system_instruction,
                tools=tools,
                token_count=token_count,
            )

    def invalidate(self, agent_id: str) -> None:
        """Clear an agent's cache state. Called on config drift, history
        clear, or tool registration."""
        names_to_delete: list[str] = []
        with self._lock:
            slot = self._slots.get(agent_id)
            if slot is None:
                return
            names_to_delete = self._collect_cache_names(slot)
            self._reset_slot_locked(slot)

        for name in names_to_delete:
            self._delete_cache(name)

    def close(self) -> None:
        """Shutdown: delete all remote caches, stop executor and reaper."""
        self._reaper_stop.set()

        all_names: list[str] = []
        with self._lock:
            self._closed = True
            for slot in self._slots.values():
                all_names.extend(self._collect_cache_names(slot, wait_for_pending=True))
            self._slots.clear()

        for name in all_names:
            self._delete_cache(name)

        self._executor.shutdown(wait=False)
        self._reaper_thread.join(timeout=2)
        logger.debug("ContextCacheRegistry closed")

    # ------------------------------------------------------------------
    # Internal: promotion
    # ------------------------------------------------------------------

    def _try_promote(self, slot: _CacheSlot) -> bool:
        """Promote pending -> ready if done. Must be called with _lock held.
        Returns True if promoted."""
        if slot.pending is None or not slot.pending.done():
            return False

        old_name = slot.ready_name
        promoted = False
        try:
            new_name = slot.pending.result(timeout=0)
            slot.ready_name = new_name
            slot.ready_offset = slot.pending_through_index or 0
            slot.ready_created_at = time.monotonic()
            promoted = True
            logger.debug(
                "Cache promoted: %s (covers %d items)", new_name, slot.ready_offset
            )
        except Exception as e:
            logger.warning("Cache creation failed: %s", e)

        slot.pending = None
        slot.pending_through_index = None

        # Schedule old cache deletion outside lock via executor
        if old_name:
            self._executor.submit(self._delete_cache, old_name)

        return promoted

    # ------------------------------------------------------------------
    # Internal: TTL
    # ------------------------------------------------------------------

    def _is_expired(self, slot: _CacheSlot) -> bool:
        """Check if the ready cache has exceeded TTL."""
        if slot.ready_created_at is None:
            return False
        return (time.monotonic() - slot.ready_created_at) >= self._cache_ttl

    def _reaper_loop(self) -> None:
        """Background thread that cleans up expired caches every 60s."""
        while not self._reaper_stop.wait(timeout=60):
            expired_names: list[str] = []
            with self._lock:
                for agent_id, slot in self._slots.items():
                    if slot.ready_name and self._is_expired(slot):
                        logger.debug(
                            "Reaper: expiring cache for %s", agent_id
                        )
                        expired_names.append(slot.ready_name)
                        self._clear_ready(slot)

            for name in expired_names:
                self._delete_cache(name)

    # ------------------------------------------------------------------
    # Internal: cache creation decisions
    # ------------------------------------------------------------------

    def _should_cache(self, slot: _CacheSlot, token_count: int) -> bool:
        """Check if a new cache creation is warranted.
        token_count must already be resolved. Must be called with _lock held."""
        if slot.pending is not None:
            return False

        if token_count < slot.min_token_threshold:
            return False

        delta = token_count - slot.last_cache_token_count
        if slot.last_cache_token_count > 0 and delta < MIN_TOKEN_GROWTH:
            return False

        return True

    def _convert_tools(self, tools: list[Callable]) -> list[types.Tool]:
        """Convert Python callables to Tool declarations for the caching API."""
        declarations = [
            types.FunctionDeclaration.from_callable(
                callable=f, client=self._client._api_client
            )
            for f in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def _fire_creation(
        self,
        slot: _CacheSlot,
        agent_id: str,
        contents: list[types.Content],
        system_instruction: str | None,
        tools: list[Callable] | None,
        token_count: int,
    ) -> None:
        """Submit background cache creation. Must be called with _lock held."""
        slot.pending_through_index = len(contents)
        slot.last_cache_token_count = token_count
        slot.config_fingerprint = _compute_fingerprint(system_instruction, tools)

        model_name = slot.model_name
        api_ttl = self._api_ttl
        client = self._client
        # Convert tools upfront (fast, in-memory) so the background
        # thread sends serialized declarations, not raw callables.
        converted_tools = self._convert_tools(tools) if tools else None

        def _create() -> str:
            cache = client.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(
                    contents=contents,
                    system_instruction=system_instruction,
                    tools=converted_tools,
                    ttl=api_ttl,
                ),
            )
            logger.debug("Cache created for %s: %s", agent_id, cache.name)
            return cache.name

        slot.pending = self._executor.submit(_create)

    # ------------------------------------------------------------------
    # Internal: slot cleanup helpers
    # ------------------------------------------------------------------

    def _collect_cache_names(
        self, slot: _CacheSlot, wait_for_pending: bool = False
    ) -> list[str]:
        """Collect cache names from a slot for deletion.

        Args:
            slot: The cache slot to drain.
            wait_for_pending: If True, wait up to 5s for an in-flight
                pending future so we can delete the resulting cache.
                If False, just cancel it.
        Must be called with _lock held.
        """
        names: list[str] = []
        if slot.pending is not None:
            if slot.pending.done():
                try:
                    names.append(slot.pending.result(timeout=0))
                except Exception:
                    pass
            else:
                slot.pending.cancel()
                if wait_for_pending:
                    try:
                        names.append(slot.pending.result(timeout=5))
                    except Exception:
                        pass
        if slot.ready_name:
            names.append(slot.ready_name)
        return names

    @staticmethod
    def _clear_ready(slot: _CacheSlot) -> None:
        """Clear the ready cache fields on a slot."""
        slot.ready_name = None
        slot.ready_offset = 0
        slot.ready_created_at = None

    def _reset_slot_locked(self, slot: _CacheSlot) -> None:
        """Reset a slot's entire cache state without removing it from the registry.
        Must be called with _lock held."""
        self._clear_ready(slot)
        slot.pending = None
        slot.pending_through_index = None
        slot.last_cache_token_count = 0
        slot.config_fingerprint = None

    def _delete_cache(self, name: str) -> None:
        """Best-effort remote cache deletion."""
        try:
            self._client.caches.delete(name=name)
            logger.debug("Cache deleted: %s", name)
        except Exception as e:
            logger.debug("Cache deletion failed (API TTL will handle): %s", e)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _compute_fingerprint(
    system_instruction: str | None, tools: list[Callable] | None
) -> str:
    """Hash system_instruction + tool names to detect config drift."""
    parts = [system_instruction or ""]
    if tools:
        parts.extend(getattr(f, "__name__", str(f)) for f in tools)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
