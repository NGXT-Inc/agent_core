"""OpenRouter provider helpers.

OpenRouter speaks the OpenAI chat-completions API, but it also exposes
provider-side prompt caching and an optional response cache through
OpenRouter-specific request fields.  This module keeps those controls
first-class while reusing ``OpenAIProvider`` for message handling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from agent_core.providers.openai import OpenAIProvider


@dataclass(frozen=True, slots=True)
class OpenRouterCacheConfig:
    """Cache controls understood by ``OpenRouterProvider``.

    Prompt caching for Moonshot/Kimi, DeepSeek, OpenAI, Grok, and some other
    providers is automatic on OpenRouter when the selected route supports it.
    This config exposes the explicit controls OpenRouter accepts:

    - ``response_cache`` enables OpenRouter's beta response cache for identical
      requests.
    - ``prompt_cache_control`` maps to the top-level ``cache_control`` request
      body field for providers that require explicit prompt cache control.
    """

    response_cache: bool | None = None
    response_cache_ttl_seconds: int | None = None
    response_cache_clear: bool = False
    prompt_cache_control: dict[str, Any] | None = None

    def to_openai_cache_config(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.response_cache is not None:
            data["response_cache"] = self.response_cache
        if self.response_cache_ttl_seconds is not None:
            data["response_cache_ttl_seconds"] = self.response_cache_ttl_seconds
        if self.response_cache_clear:
            data["response_cache_clear"] = True
        if self.prompt_cache_control:
            data["prompt_cache_control"] = self.prompt_cache_control
        return data


class OpenRouterProvider(OpenAIProvider):
    """OpenAI-compatible provider configured for OpenRouter.

    Args:
        client: Optional preconfigured ``openai.OpenAI`` compatible client.
        api_key: OpenRouter API key. If omitted, ``OPENROUTER_API_KEY`` is used.
        base_url: OpenRouter API base URL.
        app_url: Optional ``HTTP-Referer`` attribution header.
        app_name: Optional ``X-Title`` attribution header.
        response_cache: Enable OpenRouter response caching for identical
            requests. Defaults to ``False`` because it replays stale responses
            until TTL expiry.
        response_cache_ttl_seconds: Optional response-cache TTL in seconds.
        prompt_cache_control: Optional top-level ``cache_control`` object for
            explicit provider prompt caching.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        app_url: str | None = None,
        app_name: str | None = None,
        preserve_reasoning: bool = True,
        extra_body: dict | None = None,
        extra_headers: dict | None = None,
        cache_config: OpenRouterCacheConfig | dict | None = None,
        response_cache: bool | None = None,
        response_cache_ttl_seconds: int | None = None,
        prompt_cache_control: dict[str, Any] | None = None,
    ) -> None:
        if client is None:
            api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY must be set in environment or passed as api_key"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenRouterProvider requires the optional 'openai' dependency. "
                    "Install with agent-core[openai]."
                ) from exc
            client = OpenAI(base_url=base_url, api_key=api_key)

        headers = dict(extra_headers or {})
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_name:
            headers["X-Title"] = app_name

        cache_defaults: dict[str, Any] = {}
        if cache_config is not None:
            if hasattr(cache_config, "to_openai_cache_config"):
                cache_defaults.update(cache_config.to_openai_cache_config())
            else:
                cache_defaults.update(dict(cache_config))
        if response_cache is not None:
            cache_defaults["response_cache"] = response_cache
        if response_cache_ttl_seconds is not None:
            cache_defaults["response_cache_ttl_seconds"] = response_cache_ttl_seconds
        if prompt_cache_control is not None:
            cache_defaults["prompt_cache_control"] = prompt_cache_control

        super().__init__(
            client=client,
            preserve_reasoning=preserve_reasoning,
            extra_body=extra_body,
            extra_headers=headers,
            cache_config=cache_defaults or None,
        )
