"""Shared helpers for ephemeral attachment metadata."""

from __future__ import annotations


def format_attachment_placeholder(
    *,
    mime_type: str,
    label: str | None = None,
) -> str:
    """Describe an attachment whose bytes are not retained in history."""
    label = label or "unnamed attachment"
    mime_type = mime_type or "application/octet-stream"
    return f"[Attached file omitted after reload: {label} ({mime_type})]"
