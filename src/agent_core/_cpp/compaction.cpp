#include "agent_core/compaction.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace agent_core {

std::size_t select_preserved_tail_start(
    const std::vector<std::pair<std::string, std::string>>& role_and_json,
    std::uint32_t tail_token_budget, std::uint32_t min_messages,
    std::uint32_t max_chars) {
  if (role_and_json.empty()) return 0;

  std::uint32_t used_tokens = 0;
  std::size_t kept = 0;
  std::size_t start = role_and_json.size();

  while (start > 0) {
    const auto& [role, json] = role_and_json[start - 1];
    // Match the Python rendering: "<ROLE>: <content (truncated to max_chars)>"
    // Token budget is computed against this rendered string, not the raw JSON,
    // to stay in sync with the previous Python behavior.
    std::string truncated = json;
    if (truncated.size() > max_chars) {
      truncated.resize(max_chars);
      truncated.append("\n... [truncated]");
    }
    std::string upper_role = role;
    std::transform(upper_role.begin(), upper_role.end(), upper_role.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    std::string rendered = upper_role + ": " + truncated;

    auto msg_tokens = approximate_tokens(rendered);
    if (kept >= min_messages && used_tokens + msg_tokens > tail_token_budget) {
      break;
    }
    --start;
    ++kept;
    used_tokens += msg_tokens;
  }
  return start;
}

std::vector<std::string> trimmed_transcript_lines(
    const std::vector<std::string>& lines, std::uint32_t max_chars) {
  std::size_t total_chars = 0;
  for (const auto& l : lines) total_chars += l.size();
  if (total_chars <= max_chars) return lines;

  std::vector<std::string> result;
  std::size_t used = 0;
  // Head: first three lines verbatim.
  const std::size_t head_count = std::min<std::size_t>(3, lines.size());
  for (std::size_t i = 0; i < head_count; ++i) {
    result.push_back(lines[i]);
    used += lines[i].size();
  }

  // Tail: walk backwards, prepend lines that fit (leaving 128 chars of slack
  // for the ellipsis line).
  std::vector<std::string> tail;
  for (std::size_t i = lines.size(); i > head_count; --i) {
    const auto& line = lines[i - 1];
    if (used + line.size() + 128 > max_chars) break;
    tail.push_back(line);
    used += line.size();
  }
  std::reverse(tail.begin(), tail.end());

  std::size_t omitted = lines.size() - head_count - tail.size();
  if (omitted > 0) {
    std::ostringstream oss;
    oss << "[... " << omitted
        << " earlier messages omitted from compaction input ...]";
    result.push_back(oss.str());
  }
  for (const auto& l : tail) result.push_back(l);
  return result;
}

}  // namespace agent_core
