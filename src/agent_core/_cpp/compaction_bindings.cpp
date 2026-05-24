// Python bindings for the C++ compaction helpers.
//
// The Python ``agents/compaction.py`` is kept as the public surface; this
// binding lets the Python helpers delegate the hot math to C++ without
// touching the agent loop's orchestration code.

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "agent_core/compaction.hpp"

namespace nb = nanobind;

namespace agent_core {

void register_compaction_bindings(nb::module_& m) {
  m.def("approximate_tokens",
        [](const std::string& text) { return approximate_tokens(text); },
        nb::arg("text"),
        "Cheap token estimate: max(1, len/4) for non-empty input.");

  m.def("estimate_history_tokens",
        [](const std::vector<std::string>& canonicals) {
          return estimate_history_tokens(canonicals);
        },
        nb::arg("canonicals"),
        "Sum of approximate_tokens across the provided canonical JSON strings.");

  m.def(
      "select_preserved_tail_start",
      [](const std::vector<std::pair<std::string, std::string>>& role_and_json,
         std::uint32_t tail_token_budget, std::uint32_t min_messages,
         std::uint32_t max_chars) {
        return select_preserved_tail_start(role_and_json, tail_token_budget,
                                           min_messages, max_chars);
      },
      nb::arg("role_and_json"), nb::arg("tail_token_budget"),
      nb::arg("min_messages"), nb::arg("max_chars"),
      "Pick the index where the preserved tail starts. Walks backward, "
      "summing approx_tokens until adding the next would exceed the budget; "
      "always keeps at least min_messages messages.");

  m.def(
      "trimmed_transcript_lines",
      [](const std::vector<std::string>& lines, std::uint32_t max_chars) {
        return trimmed_transcript_lines(lines, max_chars);
      },
      nb::arg("lines"), nb::arg("max_chars"),
      "Truncate a transcript line list to fit within max_chars total — head "
      "three lines + as many tail lines as fit + one ellipsis line when "
      "content is dropped.");
}

}  // namespace agent_core
