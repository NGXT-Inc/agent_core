// Compaction primitives shared between Python and C++.
//
// Phase 5 moves the cheap math (token estimation, tail selection, transcript
// trimming) into C++. The summary model call stays in Python because it goes
// through the provider; the orchestration in Agent._maybe_compact_history
// invokes these helpers, then dispatches to the provider, then asks the
// HistoryStore to swap the prefix atomically.

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace agent_core {

// Mirror of the Python dataclass. Plain-old-data; consumers in C++ stay
// header-only so we don't ship two source files for a few constants.
struct CompactionConfig {
  bool enabled = false;
  std::uint32_t model_limit_tokens = 256'000;
  std::uint32_t trigger_tokens = 0;          // 0 → derive from limit*4/5
  std::uint32_t target_tokens = 128'000;
  std::uint32_t tail_token_budget = 64'000;
  std::uint32_t response_buffer_tokens = 32'768;
  std::uint32_t summary_max_output_tokens = 4096;
  std::uint32_t max_transcript_chars = 120'000;
  std::uint32_t max_message_chars = 12'000;
  std::uint32_t min_preserved_messages = 4;
  std::uint32_t max_compactions_per_run = 1;

  std::uint32_t effective_trigger_tokens() const {
    std::uint32_t trigger =
        trigger_tokens > 0 ? trigger_tokens : (model_limit_tokens * 4 / 5);
    std::uint32_t headroom =
        model_limit_tokens > response_buffer_tokens
            ? model_limit_tokens - response_buffer_tokens
            : 1;
    return std::min(trigger, std::max<std::uint32_t>(1, headroom));
  }

  std::uint32_t effective_tail_token_budget(
      std::uint32_t system_prompt_tokens = 0) const {
    std::uint32_t target_tail =
        target_tokens > system_prompt_tokens
            ? target_tokens - system_prompt_tokens
            : 1;
    return std::min(tail_token_budget, std::max<std::uint32_t>(1, target_tail));
  }
};

// Cheap approximation: max(1, (len + 3) / 4) for non-empty text, 0 otherwise.
inline std::uint32_t approximate_tokens(const std::string& text) {
  if (text.empty()) return 0;
  return static_cast<std::uint32_t>((text.size() + 3) / 4);
}

// Sum-of-approx-tokens across a vector of canonical JSON strings. Used as the
// fallback "what does the history currently cost?" estimate during the
// trigger check.
inline std::uint32_t estimate_history_tokens(
    const std::vector<std::string>& canonicals) {
  std::uint32_t total = 0;
  for (const auto& s : canonicals) total += approximate_tokens(s);
  return total;
}

// Walk a (role, canonical_json) list backwards, accumulating approx_tokens
// until we'd exceed *tail_token_budget*, then return the index where the
// preserved tail starts. Always keeps at least *min_messages* messages.
// Mirrors the existing Python ``select_preserved_tail_start`` behavior; the
// provider's adjust_compaction_tail_start is applied separately in Python.
std::size_t select_preserved_tail_start(
    const std::vector<std::pair<std::string, std::string>>& role_and_json,
    std::uint32_t tail_token_budget, std::uint32_t min_messages,
    std::uint32_t max_chars);

// Trim a transcript line list to *max_chars* total — keeps the first three
// lines and as many tail lines as fit, with a single ellipsis line filling
// in the middle when content was dropped.
std::vector<std::string> trimmed_transcript_lines(
    const std::vector<std::string>& lines, std::uint32_t max_chars);

}  // namespace agent_core
