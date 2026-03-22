"""Pipelined Vertex AI context caching for agent conversations.

Caches conversation history in the background so that subsequent
generate_content calls pay only 10% of input token cost on the
cached prefix. The pipeline runs one step behind: at round i,
the ready cache covers through round i-2. Cache creation for
round i-1 runs in the background during round i.

This module is internal to agent_core. Developers using the package
do not interact with it directly.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class CachePipeline:
    """Two-slot pipeline: pending (being created) and ready (usable).

    Lifecycle per run() call:
        1. Start of run(): if ready cache exists, use it
        2. End of run(): promote pending → ready, fire new pending
    """

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        min_token_threshold: int = 32_768,
    ):
        self._client = client
        self._model_name = model_name
        self._min_token_threshold = min_token_threshold

        # Pipeline slots
        self._pending: Future | None = None
        self._ready_name: str | None = None

        # How many history items each cache covers
        self._pending_through_index: int | None = None
        self._cached_through_index: int = 0

    @property
    def has_ready_cache(self) -> bool:
        return self._ready_name is not None

    @property
    def ready_cache_name(self) -> str | None:
        return self._ready_name

    @property
    def cached_through_index(self) -> int:
        return self._cached_through_index

    def should_cache(self, contents: list[types.Content]) -> bool:
        """Check if contents exceed the minimum token threshold for caching."""
        try:
            response = self._client.models.count_tokens(
                model=self._model_name,
                contents=contents,
            )
            token_count = response.total_tokens
            logger.debug(
                f"Cache token check: {token_count} tokens "
                f"(threshold: {self._min_token_threshold})"
            )
            return token_count >= self._min_token_threshold
        except Exception as e:
            logger.debug(f"Token counting failed, skipping cache: {e}")
            return False

    def create_cache_async(
        self,
        contents: list[types.Content],
        system_instruction: str | None,
        tools: list[Callable] | None,
        executor: ThreadPoolExecutor,
    ) -> None:
        """Fire cache creation in the background.

        Args:
            contents: Shallow copy of history to cache.
            system_instruction: System prompt to bake into cache.
            tools: Tool functions to bake into cache.
            executor: ThreadPoolExecutor to submit the work to.
        """
        if self._pending is not None:
            logger.warning("Cache creation already pending, skipping")
            return

        self._pending_through_index = len(contents)

        def _create() -> str:
            cache = self._client.caches.create(
                model=self._model_name,
                config=types.CreateCachedContentConfig(
                    contents=contents,
                    system_instruction=system_instruction,
                    tools=tools,
                    ttl="3600s",
                ),
            )
            return cache.name

        self._pending = executor.submit(_create)

    def promote_pending(self) -> None:
        """Wait for pending cache to complete, promote it to ready.

        Deletes the old ready cache. On failure, clears pending and
        leaves ready unchanged.
        """
        if self._pending is None:
            return

        old_name = self._ready_name
        try:
            new_name = self._pending.result(timeout=120)
            self._ready_name = new_name
            self._cached_through_index = self._pending_through_index or 0
            logger.debug(
                f"Cache promoted: {new_name} "
                f"(covers {self._cached_through_index} history items)"
            )
        except Exception as e:
            logger.warning(f"Cache creation failed, continuing without: {e}")

        self._pending = None
        self._pending_through_index = None

        if old_name:
            self._delete_cache(old_name)

    def invalidate(self) -> None:
        """Clear all cache state. Called on history clear or prompt change."""
        old_name = self._ready_name
        self._ready_name = None
        self._cached_through_index = 0
        self._pending = None
        self._pending_through_index = None

        if old_name:
            self._delete_cache(old_name)

    def _delete_cache(self, name: str) -> None:
        """Best-effort cache deletion. Failures are swallowed (TTL handles cleanup)."""
        try:
            self._client.caches.delete(name=name)
            logger.debug(f"Cache deleted: {name}")
        except Exception as e:
            logger.debug(f"Cache deletion failed (will expire via TTL): {e}")

    def cleanup(self) -> None:
        """Clean up all cache resources. Called on agent teardown."""
        if self._pending is not None:
            try:
                name = self._pending.result(timeout=5)
                self._delete_cache(name)
            except Exception:
                pass
            self._pending = None

        if self._ready_name:
            self._delete_cache(self._ready_name)
            self._ready_name = None

        self._cached_through_index = 0
        self._pending_through_index = None
