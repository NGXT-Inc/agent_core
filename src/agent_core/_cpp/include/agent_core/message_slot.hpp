// Internal value type for one slot in a HistoryStore.
//
// Each slot keeps the provider's native Python message object (so we can hand
// it back to the provider unchanged on the next generate call) alongside the
// canonical JSON form that gets persisted and that compaction operates on.
//
// The Python object handle is owned via nanobind's `object`, which manages
// refcount under the GIL automatically.

#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace nb = nanobind;

namespace agent_core {

// One conversation message inside a HistoryStore. The struct is intentionally
// small and trivially-movable so vector growth stays cheap. The Python object
// handle owns one strong ref to the provider-native message.
struct MessageSlot {
  // Provider-native message object — the actual SDK type the agent loop hands
  // back to provider.generate(). Empty after deserialization from SQLite until
  // the Python wrapper rebuilds it via provider.from_canonical().
  nb::object provider_native;

  // Provider-neutral serialized form. Stable across process restarts.
  std::string canonical_json;

  // Cached role from the canonical form: "user" | "assistant" | "tool" | "system".
  std::string role;

  // "gemini" | "openai" — for routing back through the right provider on reload.
  std::string provider_tag;

  // Cheap precomputed token estimate (len/4 of canonical_json). Updated when
  // the slot is rebuilt; never recomputed from the live Python object.
  std::uint32_t approx_tokens = 0;

  MessageSlot() = default;
  MessageSlot(nb::object native,
              std::string canonical,
              std::string slot_role,
              std::string tag,
              std::uint32_t tokens)
      : provider_native(std::move(native)),
        canonical_json(std::move(canonical)),
        role(std::move(slot_role)),
        provider_tag(std::move(tag)),
        approx_tokens(tokens) {}
};

}  // namespace agent_core
