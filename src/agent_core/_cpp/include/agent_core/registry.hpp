// Process-global session registry.
//
// The Registry is the single source of truth for which sessions exist in this
// process. It is accessed via Registry::instance(); any Python script in the
// process sees the same map.
//
// Session identifiers form a tree implicitly through the ":" separator —
// "user-42:designer-7f3a" is a descendant of "user-42". Subtree operations
// (cancel, list) work as prefix scans over the map.
//
// Lifecycle:
//   * acquire() returns a refcounted handle. If the session_id was previously
//     evicted but persisted, it is resurrected from SQLite on acquire.
//   * release() drops the refcount. When the refcount hits zero the session
//     becomes eligible for the reaper, which evicts it after an idle TTL.
//   * The SQLite row survives eviction — re-acquiring restores history.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "agent_core/session.hpp"

namespace agent_core {

class Registry {
 public:
  static Registry& instance();

  // Returns a session record, creating it (or resurrecting from disk) if
  // necessary. Bumps refcount; caller is responsible for matching release().
  std::shared_ptr<Session> acquire(SessionConfig cfg);

  // Look up an existing session without bumping refcount; returns null if
  // no such session is resident. Used by introspection paths.
  std::shared_ptr<Session> get(const std::string& session_id) const;

  // Release one reference; when refcount hits zero the session is left
  // resident but becomes a reaper candidate.
  void release(const std::string& session_id);

  // Set the cancellation atomic on every session whose id is *root* or
  // starts with ``root + ":"``.
  std::size_t cancel_subtree(const std::string& root);

  // Clear the cancellation atomic on a session and (optionally) its descendants.
  void clear_cancellation(const std::string& session_id, bool recursive);

  // Enumerate every resident session id.
  std::vector<std::string> list_active() const;

  // Enumerate every resident descendant of *root* (excluding root itself
  // unless ``include_root=true``).
  std::vector<std::string> descendants(const std::string& root,
                                       bool include_root = false) const;

  // Idle session reaper config. Sessions with ref_count == 0 whose last
  // access was longer than this ago get evicted by the background thread.
  void set_idle_ttl_seconds(std::uint32_t seconds);
  std::uint32_t idle_ttl_seconds() const;

  // Test-only: force the reaper to run a single sweep immediately.
  std::size_t reap_now();

  // Test-only: drop every session unconditionally. Does not touch SQLite.
  void clear();

  // Shut the background reaper down. Called from module teardown.
  void shutdown();

  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;

 private:
  Registry();
  ~Registry();

  void reaper_loop();
  static bool is_descendant(const std::string& sid, const std::string& root);

  mutable std::shared_mutex mu_;
  std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
  std::atomic<std::uint32_t> idle_ttl_seconds_{1800};  // 30 min default

  std::mutex reaper_mu_;
  std::condition_variable reaper_cv_;
  bool reaper_stop_ = false;
  std::thread reaper_thread_;
};

}  // namespace agent_core
