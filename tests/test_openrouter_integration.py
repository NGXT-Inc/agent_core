"""Opt-in live OpenRouter smoke tests.

These tests are skipped by default because they make paid API calls. Enable
with both OPENROUTER_API_KEY and RUN_OPENROUTER_INTEGRATION=1.
"""

import os

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("RUN_OPENROUTER_INTEGRATION") != "1",
    reason="set OPENROUTER_API_KEY and RUN_OPENROUTER_INTEGRATION=1",
)


@pytest.mark.parametrize(
    "model",
    [
        "moonshotai/kimi-k2.6",
        "deepseek/deepseek-v4-pro",
    ],
)
def test_openrouter_response_cache_smoke(model):
    pytest.importorskip("openai")

    from agent_core.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider(
        response_cache=True,
        response_cache_ttl_seconds=300,
        app_name="agent-core-test",
    )
    messages = [
        provider.build_user_message(
            "Return exactly these three words and nothing else: agent core cache"
        )
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "system_prompt": "You are a concise test assistant.",
        "temperature": 0,
        "max_output_tokens": 20,
        "tool_schemas": None,
    }

    first = provider.parse_response(provider.generate(**kwargs))
    second = provider.parse_response(provider.generate(**kwargs))

    assert "agent" in (first.text or "").lower()
    assert "agent" in (second.text or "").lower()
    # OpenRouter response-cache hits report zero billable usage. If this fails,
    # the response cache may have been unavailable or disabled for the account.
    assert second.usage.prompt_tokens == 0
    assert second.usage.completion_tokens == 0
