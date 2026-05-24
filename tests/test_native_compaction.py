"""Tests for the C++ compaction helpers exposed via ``agent_core._native``.

These verify byte-for-byte parity with the previous pure-Python helpers and
exercise the bindings directly so any divergence in the math is caught here
before downstream callers see it.
"""

from __future__ import annotations

from agent_core import _native


class TestApproximateTokens:
    def test_empty_returns_zero(self):
        assert _native.approximate_tokens("") == 0

    def test_one_to_four_chars_is_one_token(self):
        assert _native.approximate_tokens("a") == 1
        assert _native.approximate_tokens("abcd") == 1

    def test_five_to_eight_chars_is_two_tokens(self):
        assert _native.approximate_tokens("abcde") == 2
        assert _native.approximate_tokens("abcdefgh") == 2

    def test_ceiling_division(self):
        assert _native.approximate_tokens("a" * 9) == 3
        assert _native.approximate_tokens("a" * 100) == 25


class TestEstimateHistoryTokens:
    def test_empty_list(self):
        assert _native.estimate_history_tokens([]) == 0

    def test_sum_of_each(self):
        # "abcd" → 1 token, "abcdefgh" → 2 tokens; total = 3.
        assert _native.estimate_history_tokens(["abcd", "abcdefgh"]) == 3


class TestSelectPreservedTailStart:
    def test_empty_returns_zero(self):
        assert (
            _native.select_preserved_tail_start([], 1000, 4, 1000) == 0
        )

    def test_keeps_at_least_min_messages(self):
        # Each message is large but min_messages=2 forces us to keep them.
        msgs = [
            ("USER", "x" * 500),
            ("ASSISTANT", "y" * 500),
            ("USER", "z" * 500),
        ]
        start = _native.select_preserved_tail_start(msgs, 1, 2, 1000)
        # min_messages=2 means we must keep at least 2 from the tail.
        assert start == 1

    def test_stops_when_budget_exceeded(self):
        msgs = [
            ("USER", "a" * 8),  # 2 tokens after rendering
            ("ASSISTANT", "b" * 8),
            ("USER", "c" * 8),
        ]
        # Tiny budget: stop as soon as min_messages is met.
        start = _native.select_preserved_tail_start(msgs, 1, 1, 1000)
        assert start == 2

    def test_full_keep_when_budget_large(self):
        msgs = [("USER", "x"), ("ASSISTANT", "y"), ("USER", "z")]
        start = _native.select_preserved_tail_start(msgs, 10_000, 1, 1000)
        assert start == 0


class TestTrimmedTranscriptLines:
    def test_short_input_passes_through(self):
        lines = ["a", "b", "c"]
        assert _native.trimmed_transcript_lines(lines, 1000) == lines

    def test_long_input_keeps_head_and_tail(self):
        lines = [f"line-{i}" + "x" * 200 for i in range(10)]
        out = _native.trimmed_transcript_lines(lines, 500)
        # First three lines preserved.
        assert out[:3] == lines[:3]
        # Some "earlier messages omitted" marker is present.
        assert any("omitted from compaction input" in line for line in out)
