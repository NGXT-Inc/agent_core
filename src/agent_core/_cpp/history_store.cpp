#include "agent_core/history_store.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "agent_core/sqlite_writer.hpp"

namespace agent_core {

HistoryStore::HistoryStore(std::string session_id,
                           std::string agent_type,
                           std::string db_path)
    : session_id_(std::move(session_id)),
      agent_type_(std::move(agent_type)),
      db_path_(std::move(db_path)) {
  // If persistent, pull any existing rows. The Python wrapper is responsible
  // for re-attaching provider_native references via from_canonical() — at
  // this layer we just have the JSON text.
  if (!db_path_.empty()) {
    // Release the GIL while doing SQLite I/O — we acquired it implicitly when
    // the Python constructor called us.
    std::vector<std::string> rows;
    {
      nb::gil_scoped_release release;
      rows = SqliteWriter::instance().load(db_path_, session_id_, agent_type_);
    }
    // We need the GIL held to default-construct nb::none(); the Python
    // wrapper will replace these placeholders with real provider_native refs
    // after calling provider.from_canonical() on each canonical_json.
    slots_.reserve(rows.size());
    for (auto& json : rows) {
      MessageSlot slot;
      slot.provider_native = nb::none();
      slot.canonical_json = std::move(json);
      slot.approx_tokens =
          static_cast<std::uint32_t>(std::max<std::size_t>(
              1, (slot.canonical_json.size() + 3) / 4));
      slots_.push_back(std::move(slot));
      total_tokens_ += slots_.back().approx_tokens;
    }
  }
}

HistoryStore::~HistoryStore() = default;

void HistoryStore::append(MessageSlot slot) {
  std::uint32_t added;
  {
    std::unique_lock<std::shared_mutex> lk(mu_);
    added = slot.approx_tokens;
    slots_.push_back(std::move(slot));
    total_tokens_ += added;
    enqueue_save_locked();
  }
  (void)added;
}

void HistoryStore::replace_prefix(std::size_t prefix_len, MessageSlot summary) {
  std::unique_lock<std::shared_mutex> lk(mu_);
  if (prefix_len == 0) return;
  if (prefix_len > slots_.size()) prefix_len = slots_.size();

  std::uint32_t removed_tokens = 0;
  for (std::size_t i = 0; i < prefix_len; ++i) {
    removed_tokens += slots_[i].approx_tokens;
  }
  total_tokens_ -= removed_tokens;

  std::uint32_t added_tokens = summary.approx_tokens;
  // Replace the prefix in-place: erase the first prefix_len then prepend
  // the summary. Use a temporary vector to avoid quadratic shifts in the
  // common compaction case where prefix_len is large.
  std::vector<MessageSlot> next;
  next.reserve(slots_.size() - prefix_len + 1);
  next.push_back(std::move(summary));
  for (std::size_t i = prefix_len; i < slots_.size(); ++i) {
    next.push_back(std::move(slots_[i]));
  }
  slots_ = std::move(next);
  total_tokens_ += added_tokens;
  enqueue_save_locked();
}

void HistoryStore::clear() {
  std::unique_lock<std::shared_mutex> lk(mu_);
  slots_.clear();
  total_tokens_ = 0;
  enqueue_clear_locked();
}

void HistoryStore::rehydrate(std::vector<MessageSlot> slots) {
  std::unique_lock<std::shared_mutex> lk(mu_);
  total_tokens_ = 0;
  for (auto& s : slots) total_tokens_ += s.approx_tokens;
  slots_ = std::move(slots);
  // No enqueue: rehydrate is a load-time operation, not a mutation by the
  // user. The DB already holds the canonical snapshot.
}

void HistoryStore::flush() {
  if (db_path_.empty()) return;
  // Submit current state then drain the writer.
  {
    std::shared_lock<std::shared_mutex> lk(mu_);
    enqueue_save_locked();
  }
  nb::gil_scoped_release release;
  SqliteWriter::instance().flush_all();
}

std::vector<nb::object> HistoryStore::snapshot_native() const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  std::vector<nb::object> out;
  out.reserve(slots_.size());
  for (const auto& slot : slots_) {
    out.push_back(slot.provider_native);
  }
  return out;
}

std::vector<std::string> HistoryStore::snapshot_canonical() const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  std::vector<std::string> out;
  out.reserve(slots_.size());
  for (const auto& slot : slots_) {
    out.push_back(slot.canonical_json);
  }
  return out;
}

std::vector<std::pair<std::string, std::string>>
HistoryStore::snapshot_role_and_json() const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  std::vector<std::pair<std::string, std::string>> out;
  out.reserve(slots_.size());
  for (const auto& slot : slots_) {
    out.emplace_back(slot.role, slot.canonical_json);
  }
  return out;
}

nb::object HistoryStore::get_native(std::size_t i) const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  if (i >= slots_.size()) {
    throw std::out_of_range("HistoryStore index out of range");
  }
  return slots_[i].provider_native;
}

std::size_t HistoryStore::size() const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  return slots_.size();
}

std::uint32_t HistoryStore::total_approx_tokens() const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  return total_tokens_;
}

void HistoryStore::enqueue_save_locked() const {
  if (db_path_.empty()) return;
  WriteRequest req;
  req.db_path = db_path_;
  req.session_id = session_id_;
  req.agent_type = agent_type_;
  req.canonical_messages.reserve(slots_.size());
  for (const auto& slot : slots_) {
    req.canonical_messages.push_back(slot.canonical_json);
  }
  // We're inside a unique/shared lock on mu_; submit() takes its own lock.
  // SqliteWriter is a separate subsystem so no inversion.
  nb::gil_scoped_release release;
  SqliteWriter::instance().submit(std::move(req));
}

void HistoryStore::enqueue_clear_locked() const {
  if (db_path_.empty()) return;
  WriteRequest req;
  req.db_path = db_path_;
  req.session_id = session_id_;
  req.agent_type = agent_type_;
  req.clear = true;
  nb::gil_scoped_release release;
  SqliteWriter::instance().submit(std::move(req));
}

}  // namespace agent_core
