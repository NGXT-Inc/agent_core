// Process-global cache slot manager for Vertex AI context caches.
//
// The CacheManager owns the per-agent slot state machine — fingerprinting,
// TTL enforcement, atomic promotion of pending caches, and background work
// for cache creation/deletion. The actual Vertex API calls (create, delete)
// happen in Python via callbacks the wrapper registers at construction.
//
// The manager is keyed by an opaque ``agent_id`` string so it works with both
// the legacy Agent-instance-id path and the session-id path introduced in
// Phase 3. The Python wrapper picks the keying strategy.

#pragma once

#include <nanobind/nanobind.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace agent_core {

struct CacheAdvice {
  // Empty string ⇔ "no cache yet, use base config". Non-empty ⇔ use cached_content
  // and send contents[contents_offset:].
  std::string cache_name;
  std::size_t contents_offset = 0;
};

// Snapshot used by Python introspection so tests can verify state transitions.
struct CacheSlotSnapshot {
  std::string model_name;
  std::uint32_t min_token_threshold;
  std::string ready_name;              // empty → no ready cache
  std::size_t ready_offset;
  bool has_ready;
  bool ready_expired;
  bool pending;
  std::size_t pending_through_index;
  std::uint32_t last_cache_token_count;
  std::string config_fingerprint;
};

class CacheManager {
 public:
  static CacheManager& instance();

  // Python callbacks for the Vertex API. ``payload`` is whatever Python passed
  // to ``notify``; the manager treats it as opaque. The callback must return
  // the new cache_name (e.g. ``"cachedContents/abc-123"``). Empty string means
  // creation failed.
  using CreatePayload = nb::object;
  using CreateCallback = std::function<std::string(CreatePayload)>;
  using DeleteCallback = std::function<void(std::string cache_name)>;

  void configure(CreateCallback create, DeleteCallback del,
                 std::uint32_t max_workers, std::uint32_t cache_ttl_seconds);

  // Minimum token growth between successive caches; matches the existing
  // Python constant MIN_TOKEN_GROWTH (= 4096).
  static constexpr std::uint32_t kMinTokenGrowth = 4096;

  void register_agent(std::string agent_id, std::string model_name,
                      std::uint32_t min_token_threshold);

  // Drop the slot and synchronously delete every cache associated with it
  // (waiting up to a short timeout for any in-flight pending creation).
  void unregister_agent(const std::string& agent_id);

  // Read the best advice for ``agent_id``. If ``wait`` is true and a pending
  // creation is in flight, block up to ``wait_timeout_seconds`` for it to
  // finish before returning.
  CacheAdvice get_advice(const std::string& agent_id,
                         const std::string& fingerprint, bool wait,
                         double wait_timeout_seconds);

  // Inform the manager that the agent's contents grew by *token_count* total
  // tokens. If the slot's policy is satisfied, a background cache creation is
  // queued with ``payload`` handed to the create callback. The payload must
  // carry whatever the callback needs (model, contents snapshot, system
  // prompt, tools) — the manager does not introspect it.
  void notify(const std::string& agent_id, const std::string& fingerprint,
              std::uint32_t token_count, std::size_t contents_size,
              CreatePayload payload);

  // Clear every cache for ``agent_id`` and any in-flight pending.
  void invalidate(const std::string& agent_id);

  // Test-only: return a snapshot of the slot or std::nullopt if not registered.
  std::optional<CacheSlotSnapshot> peek_slot(const std::string& agent_id);

  // Test-only: directly seed slot state (used by tests that previously
  // constructed and mutated _CacheSlot dataclasses in Python).
  void seed_slot(const std::string& agent_id, std::string ready_name,
                 std::size_t ready_offset, bool ready_expired_now,
                 bool pending_done, std::string pending_cache_name,
                 std::size_t pending_through_index,
                 std::uint32_t last_cache_token_count,
                 std::string config_fingerprint);

  // Test/lifecycle helper — drops every slot without tearing down workers.
  // Used by ContextCacheRegistry.close() to release per-instance state
  // while leaving the global singleton ready for the next constructor call.
  std::size_t clear_slots();

  // Final teardown — stops workers + reaper + deletes remaining caches.
  // Called once by the module atexit handler.
  void close();
  bool is_configured() const;

 private:
  CacheManager();
  ~CacheManager();
  CacheManager(const CacheManager&) = delete;
  CacheManager& operator=(const CacheManager&) = delete;

  struct Slot {
    std::string model_name;
    std::uint32_t min_token_threshold = 0;

    std::string ready_name;
    std::size_t ready_offset = 0;
    std::chrono::steady_clock::time_point ready_created_at{};
    bool has_ready = false;

    std::shared_future<std::string> pending;
    std::size_t pending_through_index = 0;

    std::uint32_t last_cache_token_count = 0;
    std::string config_fingerprint;
  };

  struct CreateJob {
    std::string agent_id;
    CreatePayload payload;
    std::promise<std::string> promise;
  };

  void worker_loop();
  void reaper_loop();
  // Promote a completed pending future into ready. The previous ready_name
  // (if any, and different from the new one) is returned for the caller to
  // pass to ``delete_remote_async`` *after* releasing the lock.
  std::string promote_locked(Slot& slot);
  bool is_ready_expired_locked(const Slot& slot) const;
  void delete_remote_async(const std::string& cache_name);

  // Locks
  mutable std::shared_mutex mu_;
  std::unordered_map<std::string, std::unique_ptr<Slot>> slots_;

  // Configuration (set once via configure(); guarded by mu_ for safety)
  CreateCallback create_cb_;
  DeleteCallback delete_cb_;
  std::atomic<std::uint32_t> cache_ttl_seconds_{600};

  // Worker thread pool
  std::vector<std::thread> workers_;
  std::mutex work_mu_;
  std::condition_variable work_cv_;
  std::queue<CreateJob> work_queue_;
  bool stopping_ = false;

  // Reaper
  std::thread reaper_thread_;
  std::mutex reaper_mu_;
  std::condition_variable reaper_cv_;
  bool reaper_stop_ = false;
};

// Module-level helper exposed for tests and for fingerprint reuse by the
// Python wrapper. Matches the Python ``_compute_fingerprint`` from the
// legacy ContextCacheRegistry: sha256("|".join([system, *tool_names]))[:16].
std::string compute_fingerprint(const std::string& system_instruction,
                                const std::vector<std::string>& tool_names);

}  // namespace agent_core
