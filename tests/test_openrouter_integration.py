"""Opt-in live OpenRouter smoke tests.

These tests are skipped by default because they make paid API calls. Enable
with both OPENROUTER_API_KEY and RUN_OPENROUTER_INTEGRATION=1.
"""

import os
import time

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("RUN_OPENROUTER_INTEGRATION") != "1",
    reason="set OPENROUTER_API_KEY and RUN_OPENROUTER_INTEGRATION=1",
)


@pytest.mark.parametrize(
    "model_family,candidate_models",
    [
        ("kimi-k2.6", ["moonshotai/kimi-k2.6"]),
        ("openai-gpt-5-nano", ["openai/gpt-5-nano"]),
        ("anthropic-claude-haiku-4.5", ["anthropic/claude-haiku-4.5"]),
        ("xai-grok-4-fast", ["x-ai/grok-4-fast"]),
        (
            "deepseek-v4",
            [
                "deepseek/deepseek-v4-flash",
                "deepseek/deepseek-v4-flash:nitro",
                "deepseek/deepseek-v4-pro",
                "deepseek/deepseek-v4-pro:nitro",
            ],
        ),
    ],
)
def test_openrouter_response_cache_smoke(model_family, candidate_models):
    openai = pytest.importorskip("openai")

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
        "messages": messages,
        "system_prompt": "You are a concise test assistant.",
        "temperature": 0,
        "max_output_tokens": 256,
        "tool_schemas": None,
    }

    first_raw, model = _generate_with_model_fallback(
        provider, kwargs, candidate_models, openai, model_family
    )
    first = provider.parse_response(first_raw)
    second = provider.parse_response(provider.generate(**{**kwargs, "model": model}))

    assert "agent" in (first.text or "").lower()
    assert "agent" in (second.text or "").lower()
    # OpenRouter response-cache hits report zero billable usage. If this fails,
    # the response cache may have been unavailable or disabled for the account.
    assert second.usage.prompt_tokens == 0
    assert second.usage.completion_tokens == 0


def _generate_with_model_fallback(
    provider,
    kwargs,
    candidate_models,
    openai,
    model_family: str,
    attempts: int = 2,
):
    """Run a paid OpenRouter call against the first available model route."""
    errors = []
    unavailable_errors = []
    for model in candidate_models:
        for attempt in range(attempts):
            try:
                return provider.generate(**{**kwargs, "model": model}), model
            except openai.RateLimitError as exc:
                error = f"{model}: {exc}"
                errors.append(error)
                unavailable_errors.append(error)
                if attempt + 1 < attempts:
                    time.sleep(2)
                    continue
                break
            except openai.APIStatusError as exc:
                error = f"{model}: {exc}"
                if _is_openrouter_route_unavailable(exc):
                    errors.append(error)
                    unavailable_errors.append(error)
                    break
                if exc.status_code in {400, 404, 429, 503}:
                    errors.append(error)
                    break
                raise

    if errors and len(errors) == len(unavailable_errors):
        pytest.skip(
            f"No OpenRouter route is currently available for {model_family}. "
            "This is an account/provider routing issue, not an agent-core "
            "response-cache regression. Check OpenRouter Settings > Privacy "
            "or add a BYOK provider integration. "
            + " | ".join(errors)
        )

    pytest.fail(
        f"No OpenRouter route succeeded for {model_family}. "
        "Check OpenRouter Settings > Privacy for provider/data-policy guardrails, "
        "or add a BYOK provider integration for this model family. "
        + " | ".join(errors)
    )


def _is_openrouter_route_unavailable(exc) -> bool:
    """Return True for transient/account route failures outside this package."""
    message = str(exc).lower()
    if getattr(exc, "status_code", None) in {429, 503}:
        return True
    return (
        getattr(exc, "status_code", None) == 404
        and "no endpoints available" in message
        and ("guardrail" in message or "data policy" in message)
    )
