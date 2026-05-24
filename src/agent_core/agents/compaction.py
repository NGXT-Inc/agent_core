"""Conversation compaction helpers shared by agent runtimes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

DEFAULT_TRIGGER_RATIO_NUMERATOR = 4
DEFAULT_TRIGGER_RATIO_DENOMINATOR = 5


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    """Policy for automatic context compaction."""

    enabled: bool = False
    model_limit_tokens: int = 256_000
    trigger_tokens: int | None = None
    target_tokens: int = 128_000
    tail_token_budget: int = 64_000
    response_buffer_tokens: int = 32_768
    summary_max_output_tokens: int = 4096
    max_transcript_chars: int = 120_000
    max_message_chars: int = 12_000
    min_preserved_messages: int = 4
    max_compactions_per_run: int = 1

    def _default_trigger_tokens(self) -> int:
        return max(
            1,
            self.model_limit_tokens
            * DEFAULT_TRIGGER_RATIO_NUMERATOR
            // DEFAULT_TRIGGER_RATIO_DENOMINATOR,
        )

    def __post_init__(self) -> None:
        if self.trigger_tokens is None:
            object.__setattr__(self, "trigger_tokens", self._default_trigger_tokens())

    def effective_trigger_tokens(self) -> int:
        """Return the active auto-compaction threshold."""
        trigger = self.trigger_tokens or self._default_trigger_tokens()
        headroom_limit = max(1, self.model_limit_tokens - self.response_buffer_tokens)
        return min(trigger, headroom_limit)

    def effective_tail_token_budget(self, *, system_prompt_tokens: int = 0) -> int:
        """Return the preserved-tail budget after applying the target context."""
        target_tail_budget = max(1, self.target_tokens - max(0, system_prompt_tokens))
        return min(self.tail_token_budget, target_tail_budget)


def approximate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except Exception:
        return str(value)


def render_message_for_compaction(provider, message: Any, *, max_chars: int) -> str:
    rendered = provider.format_message_for_display(message)
    if rendered is not None:
        role = str(rendered.get("role") or "unknown").upper()
        content = str(rendered.get("content") or "").strip()
    else:
        serialized = provider.serialize_message(message)
        role = str(serialized.get("role") or "unknown").upper()
        pieces: list[str] = []
        for part in serialized.get("parts", []):
            part_type = part.get("type")
            if part_type == "text":
                pieces.append(part.get("text", ""))
            elif part_type == "function_call":
                pieces.append(
                    f"[Tool Call: {part.get('name')}({_stringify(part.get('args') or {})})]"
                )
            elif part_type == "function_response":
                pieces.append(
                    f"[Tool Response: {part.get('name')} -> {_stringify(part.get('response'))}]"
                )
            elif part_type == "inline_data":
                pieces.append(
                    f"[Inline Data omitted: {part.get('mime_type', 'application/octet-stream')}]"
                )
            elif part_type == "thought":
                pieces.append(part.get("thought", ""))
        content = "\n".join(piece for piece in pieces if piece).strip()

    if not content:
        content = "[empty message]"
    if len(content) > max_chars:
        content = content[:max_chars].rstrip() + "\n... [truncated]"
    return f"{role}: {content}"


def estimate_history_tokens(provider, messages: list[Any], *, max_chars: int) -> int:
    total = 0
    for message in messages:
        total += approximate_tokens(
            render_message_for_compaction(provider, message, max_chars=max_chars)
        )
    return total


def select_preserved_tail_start(
    provider,
    messages: list[Any],
    *,
    tail_token_budget: int,
    min_messages: int,
    max_chars: int,
) -> int:
    if not messages:
        return 0

    used_tokens = 0
    kept = 0
    start = len(messages)

    while start > 0:
        message = messages[start - 1]
        msg_tokens = approximate_tokens(
            render_message_for_compaction(provider, message, max_chars=max_chars)
        )
        if kept >= min_messages and used_tokens + msg_tokens > tail_token_budget:
            break
        start -= 1
        kept += 1
        used_tokens += msg_tokens

    adjust_start = getattr(provider, "adjust_compaction_tail_start", None)
    if adjust_start is not None:
        start = adjust_start(messages, start)
    return max(0, start)


def trimmed_transcript_lines(lines: list[str], *, max_chars: int) -> list[str]:
    total_chars = sum(len(line) for line in lines)
    if total_chars <= max_chars:
        return lines

    head = lines[:3]
    used = sum(len(line) for line in head)
    tail: list[str] = []
    for line in reversed(lines[3:]):
        if used + len(line) + 128 > max_chars:
            break
        tail.append(line)
        used += len(line)

    omitted = max(0, len(lines) - len(head) - len(tail))
    middle = []
    if omitted:
        middle.append(f"[... {omitted} earlier messages omitted from compaction input ...]")
    return [*head, *middle, *reversed(tail)]
