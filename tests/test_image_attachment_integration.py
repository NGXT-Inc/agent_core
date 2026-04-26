"""Opt-in live image attachment OCR comparison tests.

These tests make paid API calls. Enable with:

    RUN_IMAGE_ATTACHMENT_INTEGRATION=1 pytest -s tests/test_image_attachment_integration.py

Direct Gemini also requires GOOGLE_PROJECT_ID / GOOGLE_LOCATION or equivalent
ADC configuration. OpenRouter accepts either OPENROUTER_API_KEY or the common
OPEN_ROUTER_API_KEY alias from .env.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


load_dotenv()
if os.environ.get("OPEN_ROUTER_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.environ["OPEN_ROUTER_API_KEY"]


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_IMAGE_ATTACHMENT_INTEGRATION") != "1",
    reason="set RUN_IMAGE_ATTACHMENT_INTEGRATION=1 to run live image tests",
)


EXCERPT_IMAGE = Path(__file__).parent / "fixtures" / "excerpt.jpg"
PROMPT = """\
The attached images are overlapping crops of the same photographed page.
Transcribe ONLY the text inside the thick white-circled region. Use all crops
together, preserve reading order and line breaks, and do not summarize. If a
word is partially obscured or uncertain, write [unclear] for that word.
"""

EXPECTED_PHRASES = (
    "text books",
    "human heads",
    "unified body of work",
    "younger mathematicians",
    "navigate the existing",
    "barrier to entry",
)


def _prepare_ocr_images(tmp_path: Path) -> list[Path]:
    """Create stable OCR targets from the original page image."""
    Image = pytest.importorskip("PIL.Image")
    ImageEnhance = pytest.importorskip("PIL.ImageEnhance")
    ImageFilter = pytest.importorskip("PIL.ImageFilter")
    ImageOps = pytest.importorskip("PIL.ImageOps")

    image = Image.open(EXCERPT_IMAGE).convert("L")
    if image.size != (3000, 4000):
        pytest.skip(f"{EXCERPT_IMAGE} has unexpected dimensions: {image.size}")

    # These coordinates isolate overlapping views of the fixture's circled
    # region while preserving enough context for model OCR.
    crop_specs = [
        ("left", (100, 1550, 1600, 3700)),
        ("middle", (750, 1550, 2300, 3700)),
        ("right", (1450, 1550, 2900, 3700)),
        ("wide", (100, 1550, 2900, 3700)),
    ]

    output_paths: list[Path] = []
    for label, box in crop_specs:
        crop = image.crop(box)
        crop = ImageOps.autocontrast(crop)
        crop = ImageEnhance.Contrast(crop).enhance(1.8)
        crop = ImageEnhance.Sharpness(crop).enhance(2.0)
        crop = crop.filter(ImageFilter.SHARPEN)

        output_path = tmp_path / f"excerpt-circled-{label}.jpg"
        crop.convert("RGB").save(output_path, quality=95)
        output_paths.append(output_path)

    return output_paths


def _score_transcription(text: str) -> int:
    normalized = " ".join(text.lower().replace("\n", " ").split())
    return sum(1 for phrase in EXPECTED_PHRASES if phrase in normalized)


def _run_agent(provider_name: str, model: str, image_paths: list[Path]) -> str:
    from agent_core import Agent, FilePart, OpenRouterProvider

    attachments = [
        FilePart.from_path(
            image_path,
            mime_type="image/jpeg",
            filename=image_path.name,
        )
        for image_path in image_paths
    ]

    if provider_name == "openrouter":
        provider = OpenRouterProvider(app_name="agent-core-circled-text-smoke")
        agent = Agent(provider=provider, model_name=model)
    elif provider_name == "gemini":
        agent = Agent(model_name=model)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    return agent.run(
        PROMPT,
        attachments=attachments,
        temperature=0,
        max_output_tokens=1024,
    ).strip()


def test_excerpt_circled_text_extraction_comparison(tmp_path):
    """Compare direct Gemini and OpenRouter OCR of the circled image text."""
    if not EXCERPT_IMAGE.exists():
        pytest.skip(f"{EXCERPT_IMAGE} is required for this live smoke test")

    image_paths = _prepare_ocr_images(tmp_path)
    candidates = [
        ("gemini", "gemini-3-flash-preview"),
        ("gemini", "gemini-3.1-pro-preview"),
        ("openrouter", "google/gemini-2.5-flash"),
        ("openrouter", "anthropic/claude-haiku-4.5"),
    ]

    results: list[tuple[str, str, str, int]] = []
    errors: list[str] = []

    for provider_name, model in candidates:
        try:
            text = _run_agent(provider_name, model, image_paths)
            score = _score_transcription(text)
            results.append((provider_name, model, text, score))
            print(f"\n=== {provider_name} :: {model} :: score={score} ===")
            print(text or "[EMPTY]")
        except Exception as exc:
            errors.append(f"{provider_name}::{model}: {type(exc).__name__}: {exc}")
            print(f"\n=== {provider_name} :: {model} :: ERROR ===")
            print(f"{type(exc).__name__}: {exc}")

    assert results, "No model returned a transcription. Errors: " + "; ".join(errors)

    provider_scores: dict[str, int] = {}
    provider_outputs: dict[str, list[str]] = {}
    for provider_name, _model, _text, score in results:
        provider_scores[provider_name] = max(provider_scores.get(provider_name, 0), score)
        provider_outputs.setdefault(provider_name, []).append(_text)

    assert provider_scores.get("gemini", 0) >= 4, (
        "Direct Gemini did not extract enough expected circled text. "
        f"Errors: {errors}"
    )
    assert provider_scores.get("openrouter", 0) >= 4, (
        "OpenRouter did not extract enough expected circled text. "
        f"Errors: {errors}"
    )
