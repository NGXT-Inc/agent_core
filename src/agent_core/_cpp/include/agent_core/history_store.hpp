// In-memory conversation history backed by a vector of MessageSlot.
//
// HistoryStore is the single source of truth for one (session_id, agent_type)
// pair while the process is running. SQLite persistence runs through
// SqliteWriter; that's a side channel and the in-memory store is authoritative
// during normal operation.
//
// Concurrency: each instance protects its own state with a shared_mutex.
// Reads (snapshot_*, size, total_approx_tokens) take a shared lock; mutations
// (append, replace_prefix, clear) take a unique lock. Calls into Python (for
// snapshot_native, which returns nb::object refs) hold the GIL; pure C++
// reads release the GIL for parallelism.

#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include "agent_core/message_slot.hpp"

namespace nb = nanobind;

namespace agent_core {

class HistoryStore {
 public:
  // Construct a store for (session_id, agent_type). If db_path is non-empty,
  // mutations enqueue writes to the singleton SqliteWriter; on construction
  // any persisted state is loaded as canonical-only slots (provider_native is
  // empty until the Python wrapper repopulates it via from_canonical()).
  HistoryStore(std::string session_id,
               std::string agent_type,
               std::string db_path);
  ~HistoryStore();

  HistoryStore(const HistoryStore&) = delete;
  HistoryStore& operator=(const HistoryStore&) = delete;

  // --- Mutations ---

  // Append one message. The slot is moved into place; the caller must have
  // already canonicalized via the provider.
  void append(MessageSlot slot);

  // Replace the first ``prefix_len`` slots with a single summary slot. Used by
  // compaction to collapse old context. If prefix_len is 0 the summary becomes
  // a no-op (the store is unchanged); if prefix_len ≥ size() everything is
  // replaced.
  void replace_prefix(std::size_t prefix_len, MessageSlot summary);

  // Drop all messages.
  void clear();

  // Rebuild the entire store from canonical-only slots. Used after the Python
  // wrapper rehydrates provider_native refs for a resurrected session.
  void rehydrate(std::vector<MessageSlot> slots);

  // Persist the current state synchronously (blocks until SQLite write
  // completes). Useful for tests and explicit durability points.
  void flush();

  // --- Reads ---

  // Snapshot of every provider_native object — for sending to provider.generate.
  // Holds the GIL while copying refs.
  std::vector<nb::object> snapshot_native() const;

  // Snapshot of every canonical JSON string — for compaction transcript
  // rendering and for serialization. GIL-free.
  std::vector<std::string> snapshot_canonical() const;

  // Snapshot of (role, canonical_json) pairs — for compaction tail selection
  // that needs both pieces. GIL-free.
  std::vector<std::pair<std::string, std::string>> snapshot_role_and_json() const;

  // Single-message lookup; returns the provider_native at index ``i`` (holds
  // the GIL) or throws std::out_of_range.
  nb::object get_native(std::size_t i) const;

  std::size_t size() const;
  std::uint32_t total_approx_tokens() const;

  const std::string& session_id() const { return session_id_; }
  const std::string& agent_type() const { return agent_type_; }
  const std::string& db_path() const { return db_path_; }
  bool is_persistent() const { return !db_path_.empty(); }

 private:
  void enqueue_save_locked() const;
  void enqueue_clear_locked() const;

  mutable std::shared_mutex mu_;
  std::vector<MessageSlot> slots_;
  std::uint32_t total_tokens_ = 0;

  std::string session_id_;
  std::string agent_type_;
  std::string db_path_;
};

}  // namespace agent_core
