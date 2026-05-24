// Per-session record stored in the global Registry.
//
// One Session corresponds to one (session_id, agent_type) pair. It owns the
// in-memory history (via HistoryStore), tracks reference count for lifecycle,
// and carries the atomic cancellation flag that propagates through agent
// subtrees.
//
// Concurrency:
//   * The Session itself is reference-counted via std::atomic; multiple
//     SessionHandles can refer to the same Session concurrently.
//   * Member fields that mutate (history, cache slot, last_access_ns) are
//     protected either by the HistoryStore's own lock or by Session::mu_.
//   * The cancellation flag is a std::atomic<bool> so the agent loop can poll
//     it without locking.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <utility>

#include "agent_core/history_store.hpp"

namespace agent_core {

struct SessionConfig {
  std::string session_id;
  std::string agent_type;
  std::string db_path;
};

class Session {
 public:
  Session(SessionConfig cfg, std::shared_ptr<HistoryStore> history)
      : config_(std::move(cfg)),
        history_(std::move(history)),
        last_access_ns_(now_ns()) {}

  const std::string& session_id() const { return config_.session_id; }
  const std::string& agent_type() const { return config_.agent_type; }
  const std::string& db_path() const { return config_.db_path; }

  HistoryStore& history() { return *history_; }
  const HistoryStore& history() const { return *history_; }
  std::shared_ptr<HistoryStore> history_ptr() const { return history_; }

  // --- Cancellation ---
  bool is_cancelled() const noexcept { return cancelled_.load(std::memory_order_acquire); }
  void cancel() noexcept { cancelled_.store(true, std::memory_order_release); }
  void clear_cancellation() noexcept {
    cancelled_.store(false, std::memory_order_release);
  }

  // --- Refcount (Registry-only API) ---
  std::uint32_t inc_ref() noexcept {
    return ref_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
  }
  std::uint32_t dec_ref() noexcept {
    return ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  }
  std::uint32_t ref_count() const noexcept {
    return ref_count_.load(std::memory_order_acquire);
  }

  // --- Access bookkeeping ---
  void touch() noexcept { last_access_ns_.store(now_ns(), std::memory_order_relaxed); }
  std::uint64_t last_access_ns() const noexcept {
    return last_access_ns_.load(std::memory_order_relaxed);
  }

 private:
  static std::uint64_t now_ns() noexcept {
    using clock = std::chrono::steady_clock;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock::now().time_since_epoch())
            .count());
  }

  SessionConfig config_;
  std::shared_ptr<HistoryStore> history_;
  std::atomic<bool> cancelled_{false};
  std::atomic<std::uint32_t> ref_count_{0};
  std::atomic<std::uint64_t> last_access_ns_;
};

}  // namespace agent_core
