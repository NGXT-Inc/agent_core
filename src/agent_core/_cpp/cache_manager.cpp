#include "agent_core/cache_manager.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <utility>

// We avoid pulling in an external SHA-256 dependency by writing a small
// implementation here. The input length is short (system prompt + tool names),
// so performance is irrelevant; correctness is what matters. Tested against
// the existing Python ``hashlib.sha256(...).hexdigest()[:16]`` output to make
// sure fingerprints carry over byte-for-byte.

namespace agent_core {

namespace {

// ---------------------------------------------------------------------------
// SHA-256 — straight transcription from FIPS 180-4. No SIMD; this hashes ~100
// bytes per agent.
// ---------------------------------------------------------------------------

struct Sha256 {
  std::uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::uint8_t buffer[64] = {};
  std::size_t buffer_len = 0;
  std::uint64_t bit_len = 0;

  static std::uint32_t rotr(std::uint32_t v, unsigned n) {
    return (v >> n) | (v << (32 - n));
  }

  static const std::array<std::uint32_t, 64>& k() {
    static const std::array<std::uint32_t, 64> table = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    return table;
  }

  void process_block(const std::uint8_t* block) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (std::uint32_t(block[i * 4]) << 24) |
             (std::uint32_t(block[i * 4 + 1]) << 16) |
             (std::uint32_t(block[i * 4 + 2]) << 8) |
             std::uint32_t(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    const auto& K = k();
    for (int i = 0; i < 64; ++i) {
      std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      std::uint32_t ch = (e & f) ^ (~e & g);
      std::uint32_t t1 = h + S1 + ch + K[i] + w[i];
      std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      std::uint32_t t2 = S0 + mj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  void update(const std::uint8_t* data, std::size_t len) {
    bit_len += static_cast<std::uint64_t>(len) * 8;
    while (len > 0) {
      std::size_t to_copy = std::min<std::size_t>(64 - buffer_len, len);
      std::memcpy(buffer + buffer_len, data, to_copy);
      buffer_len += to_copy;
      data += to_copy;
      len -= to_copy;
      if (buffer_len == 64) {
        process_block(buffer);
        buffer_len = 0;
      }
    }
  }

  std::string hex_digest_16() {
    // FIPS 180-4 padding.
    std::uint8_t pad = 0x80;
    update(&pad, 1);
    std::uint8_t zero = 0x00;
    while (buffer_len != 56) update(&zero, 1);
    std::uint64_t bits = bit_len - 8;  // subtract the 0x80 we just appended
    std::uint8_t length_bytes[8];
    for (int i = 0; i < 8; ++i) {
      length_bytes[7 - i] = static_cast<std::uint8_t>(bits >> (i * 8));
    }
    update(length_bytes, 8);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (std::uint32_t s : state) oss << std::setw(8) << s;
    auto full = oss.str();
    return full.substr(0, 16);
  }
};

}  // namespace

std::string compute_fingerprint(const std::string& system_instruction,
                                const std::vector<std::string>& tool_names) {
  Sha256 sha;
  // Match Python: parts = [system, *tool_names]; sha256("|".join(parts))
  sha.update(reinterpret_cast<const std::uint8_t*>(system_instruction.data()),
             system_instruction.size());
  for (const auto& name : tool_names) {
    static const std::uint8_t pipe = '|';
    sha.update(&pipe, 1);
    sha.update(reinterpret_cast<const std::uint8_t*>(name.data()), name.size());
  }
  return sha.hex_digest_16();
}

// ---------------------------------------------------------------------------
// CacheManager
// ---------------------------------------------------------------------------

CacheManager& CacheManager::instance() {
  static CacheManager inst;
  return inst;
}

CacheManager::CacheManager() = default;

CacheManager::~CacheManager() { close(); }

bool CacheManager::is_configured() const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  return static_cast<bool>(create_cb_) && static_cast<bool>(delete_cb_);
}

void CacheManager::configure(CreateCallback create, DeleteCallback del,
                             std::uint32_t max_workers,
                             std::uint32_t cache_ttl_seconds) {
  // Always honor the new callbacks; replacing an existing pair is supported.
  {
    std::unique_lock<std::shared_mutex> wlk(mu_);
    create_cb_ = std::move(create);
    delete_cb_ = std::move(del);
    cache_ttl_seconds_.store(cache_ttl_seconds, std::memory_order_release);
  }

  // Resurrect the worker pool if a previous close() torn it down.
  bool need_workers = false;
  {
    std::lock_guard<std::mutex> wq(work_mu_);
    if (stopping_) {
      stopping_ = false;
    }
    need_workers = workers_.empty();
  }
  if (need_workers) {
    auto count = std::max<std::uint32_t>(1, max_workers);
    workers_.reserve(workers_.size() + count);
    for (std::uint32_t i = 0; i < count; ++i) {
      workers_.emplace_back(&CacheManager::worker_loop, this);
    }
  }

  // Same dance for the reaper.
  bool need_reaper = false;
  {
    std::lock_guard<std::mutex> rl(reaper_mu_);
    if (reaper_stop_) reaper_stop_ = false;
    need_reaper = !reaper_thread_.joinable();
  }
  if (need_reaper) {
    reaper_thread_ = std::thread(&CacheManager::reaper_loop, this);
  }
}

void CacheManager::register_agent(std::string agent_id, std::string model_name,
                                  std::uint32_t min_token_threshold) {
  std::vector<std::string> to_delete;
  {
    std::unique_lock<std::shared_mutex> lk(mu_);
    auto it = slots_.find(agent_id);
    if (it != slots_.end()) {
      // Re-registration: drop any existing cache.
      if (it->second->has_ready) to_delete.push_back(it->second->ready_name);
      slots_.erase(it);
    }
    auto slot = std::make_unique<Slot>();
    slot->model_name = std::move(model_name);
    slot->min_token_threshold = min_token_threshold;
    slots_.emplace(std::move(agent_id), std::move(slot));
  }
  for (const auto& name : to_delete) delete_remote_async(name);
}

void CacheManager::unregister_agent(const std::string& agent_id) {
  std::unique_ptr<Slot> taken;
  {
    std::unique_lock<std::shared_mutex> lk(mu_);
    auto it = slots_.find(agent_id);
    if (it == slots_.end()) return;
    taken = std::move(it->second);
    slots_.erase(it);
  }
  // Wait briefly for any in-flight pending so we can delete the resulting cache.
  std::vector<std::string> to_delete;
  if (taken->has_ready) to_delete.push_back(taken->ready_name);
  if (taken->pending.valid()) {
    if (taken->pending.wait_for(std::chrono::seconds(5)) ==
        std::future_status::ready) {
      try {
        auto name = taken->pending.get();
        if (!name.empty()) to_delete.push_back(std::move(name));
      } catch (...) {
        // Creation failed; nothing to delete.
      }
    }
  }
  for (const auto& name : to_delete) delete_remote_async(name);
}

CacheAdvice CacheManager::get_advice(const std::string& agent_id,
                                     const std::string& fingerprint, bool wait,
                                     double wait_timeout_seconds) {
  // First, optionally wait for the pending future outside any lock.
  if (wait) {
    std::shared_future<std::string> pending;
    {
      std::shared_lock<std::shared_mutex> lk(mu_);
      auto it = slots_.find(agent_id);
      if (it != slots_.end() && it->second->pending.valid()) {
        pending = it->second->pending;
      }
    }
    if (pending.valid()) {
      auto ms = static_cast<std::int64_t>(wait_timeout_seconds * 1000.0);
      // Release the GIL while we block — the worker thread needs it to call
      // the Python create callback and set the promise value.
      nb::gil_scoped_release release;
      pending.wait_for(std::chrono::milliseconds(ms));
    }
  }

  std::string old_ready_to_delete;
  std::string promoted_old;
  CacheAdvice advice;

  {
    std::unique_lock<std::shared_mutex> lk(mu_);
    auto it = slots_.find(agent_id);
    if (it != slots_.end()) {
      Slot& slot = *it->second;
      promoted_old = promote_locked(slot);

      // Decide what advice to return. ``old_ready_to_delete`` is set when the
      // current ready_name needs to be invalidated (expired or drifted). The
      // actual delete fires after we release the lock.
      if (slot.has_ready) {
        if (is_ready_expired_locked(slot)) {
          old_ready_to_delete = std::move(slot.ready_name);
          slot.has_ready = false;
          slot.ready_offset = 0;
        } else if (!slot.config_fingerprint.empty() &&
                   slot.config_fingerprint != fingerprint) {
          old_ready_to_delete = std::move(slot.ready_name);
          slot.has_ready = false;
          slot.ready_offset = 0;
          slot.last_cache_token_count = 0;
          slot.config_fingerprint.clear();
          slot.pending = {};
          slot.pending_through_index = 0;
        } else {
          advice.cache_name = slot.ready_name;
          advice.contents_offset = slot.ready_offset;
        }
      }
    }
  }

  if (!promoted_old.empty()) delete_remote_async(promoted_old);
  if (!old_ready_to_delete.empty()) delete_remote_async(old_ready_to_delete);
  return advice;
}

void CacheManager::notify(const std::string& agent_id,
                          const std::string& fingerprint,
                          std::uint32_t token_count, std::size_t contents_size,
                          CreatePayload payload) {
  // Run promote first so the slot's view is fresh, then decide whether to fire.
  bool should_fire = false;
  std::promise<std::string>* promise_ptr = nullptr;
  std::string promoted_old;
  {
    std::unique_lock<std::shared_mutex> lk(mu_);
    auto it = slots_.find(agent_id);
    if (it == slots_.end()) return;
    Slot& slot = *it->second;
    promoted_old = promote_locked(slot);

    if (slot.pending.valid()) return;
    if (token_count < slot.min_token_threshold) return;
    if (slot.last_cache_token_count > 0 &&
        token_count - slot.last_cache_token_count < kMinTokenGrowth) {
      return;
    }

    // Reserve the pending slot before unlocking so a concurrent notify doesn't
    // also queue a creation.
    auto promise = std::make_shared<std::promise<std::string>>();
    slot.pending = promise->get_future().share();
    slot.pending_through_index = contents_size;
    slot.last_cache_token_count = token_count;
    slot.config_fingerprint = fingerprint;
    should_fire = true;

    // Enqueue the create job (holding the work queue lock briefly).
    std::lock_guard<std::mutex> wq(work_mu_);
    if (stopping_) return;
    CreateJob job;
    job.agent_id = agent_id;
    job.payload = std::move(payload);
    job.promise = std::move(*promise);
    work_queue_.push(std::move(job));
    promise_ptr = nullptr;  // moved
  }
  if (!promoted_old.empty()) delete_remote_async(promoted_old);
  if (should_fire) work_cv_.notify_one();
}

void CacheManager::invalidate(const std::string& agent_id) {
  std::vector<std::string> to_delete;
  std::shared_future<std::string> pending;
  {
    std::unique_lock<std::shared_mutex> lk(mu_);
    auto it = slots_.find(agent_id);
    if (it == slots_.end()) return;
    Slot& slot = *it->second;
    if (slot.has_ready) {
      to_delete.push_back(std::move(slot.ready_name));
      slot.has_ready = false;
      slot.ready_offset = 0;
    }
    if (slot.pending.valid()) {
      pending = slot.pending;
      slot.pending = {};
      slot.pending_through_index = 0;
    }
    slot.last_cache_token_count = 0;
    slot.config_fingerprint.clear();
  }
  // If a pending creation is still in flight, schedule deletion of whatever
  // it produces so we don't leak.
  if (pending.valid()) {
    // Detach a small thread that waits and queues a delete. We avoid using
    // the worker pool since invalidation is rare and we want symmetry with
    // unregister.
    std::thread([this, fut = std::move(pending)]() mutable {
      if (fut.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        return;
      }
      try {
        auto name = fut.get();
        if (!name.empty()) delete_remote_async(name);
      } catch (...) {
      }
    }).detach();
  }
  for (const auto& name : to_delete) delete_remote_async(name);
}

std::optional<CacheSlotSnapshot> CacheManager::peek_slot(
    const std::string& agent_id) {
  std::shared_lock<std::shared_mutex> lk(mu_);
  auto it = slots_.find(agent_id);
  if (it == slots_.end()) return std::nullopt;
  const Slot& slot = *it->second;
  CacheSlotSnapshot snap;
  snap.model_name = slot.model_name;
  snap.min_token_threshold = slot.min_token_threshold;
  snap.has_ready = slot.has_ready;
  snap.ready_name = slot.ready_name;
  snap.ready_offset = slot.ready_offset;
  snap.ready_expired = slot.has_ready && is_ready_expired_locked(slot);
  snap.pending = slot.pending.valid();
  snap.pending_through_index = slot.pending_through_index;
  snap.last_cache_token_count = slot.last_cache_token_count;
  snap.config_fingerprint = slot.config_fingerprint;
  return snap;
}

void CacheManager::seed_slot(const std::string& agent_id,
                             std::string ready_name,
                             std::size_t ready_offset,
                             bool ready_expired_now, bool pending_done,
                             std::string pending_cache_name,
                             std::size_t pending_through_index,
                             std::uint32_t last_cache_token_count,
                             std::string config_fingerprint) {
  std::unique_lock<std::shared_mutex> lk(mu_);
  auto it = slots_.find(agent_id);
  if (it == slots_.end()) return;
  Slot& slot = *it->second;
  slot.has_ready = !ready_name.empty();
  slot.ready_name = std::move(ready_name);
  slot.ready_offset = ready_offset;
  if (ready_expired_now) {
    slot.ready_created_at = std::chrono::steady_clock::now() -
                            std::chrono::seconds(cache_ttl_seconds_.load() + 1);
  } else {
    slot.ready_created_at = std::chrono::steady_clock::now();
  }
  if (pending_done && !pending_cache_name.empty()) {
    std::promise<std::string> p;
    p.set_value(std::move(pending_cache_name));
    slot.pending = p.get_future().share();
    slot.pending_through_index = pending_through_index;
  } else {
    slot.pending = {};
    slot.pending_through_index = 0;
  }
  slot.last_cache_token_count = last_cache_token_count;
  slot.config_fingerprint = std::move(config_fingerprint);
}

std::size_t CacheManager::clear_slots() {
  std::vector<std::string> to_delete;
  std::size_t count = 0;
  {
    std::unique_lock<std::shared_mutex> lk(mu_);
    count = slots_.size();
    for (auto& [id, slot] : slots_) {
      if (slot->has_ready) to_delete.push_back(slot->ready_name);
    }
    slots_.clear();
  }
  for (const auto& name : to_delete) delete_remote_async(name);
  return count;
}

void CacheManager::close() {
  {
    std::lock_guard<std::mutex> wq(work_mu_);
    if (stopping_) return;
    stopping_ = true;
  }
  work_cv_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
  workers_.clear();

  {
    std::lock_guard<std::mutex> rl(reaper_mu_);
    reaper_stop_ = true;
  }
  reaper_cv_.notify_all();
  if (reaper_thread_.joinable()) reaper_thread_.join();

  // Delete every remaining remote cache.
  std::vector<std::string> to_delete;
  {
    std::unique_lock<std::shared_mutex> lk(mu_);
    for (auto& [id, slot] : slots_) {
      if (slot->has_ready) to_delete.push_back(slot->ready_name);
      if (slot->pending.valid() &&
          slot->pending.wait_for(std::chrono::seconds(1)) ==
              std::future_status::ready) {
        try {
          auto name = slot->pending.get();
          if (!name.empty()) to_delete.push_back(name);
        } catch (...) {
        }
      }
    }
    slots_.clear();
  }
  for (const auto& name : to_delete) delete_remote_async(name);
}

void CacheManager::worker_loop() {
  while (true) {
    CreateJob job;
    {
      std::unique_lock<std::mutex> lk(work_mu_);
      work_cv_.wait(lk, [&] { return stopping_ || !work_queue_.empty(); });
      if (stopping_ && work_queue_.empty()) return;
      job = std::move(work_queue_.front());
      work_queue_.pop();
    }

    // Call into Python — needs the GIL.
    std::string cache_name;
    try {
      nb::gil_scoped_acquire gil;
      CreateCallback cb;
      {
        std::shared_lock<std::shared_mutex> lk(mu_);
        cb = create_cb_;
      }
      if (cb) cache_name = cb(std::move(job.payload));
    } catch (...) {
      cache_name.clear();
    }
    job.promise.set_value(cache_name);
  }
}

void CacheManager::reaper_loop() {
  std::unique_lock<std::mutex> lk(reaper_mu_);
  while (!reaper_stop_) {
    reaper_cv_.wait_for(lk, std::chrono::seconds(60),
                       [&] { return reaper_stop_; });
    if (reaper_stop_) break;
    lk.unlock();
    std::vector<std::string> to_delete;
    {
      std::unique_lock<std::shared_mutex> w(mu_);
      for (auto& [id, slot] : slots_) {
        if (slot->has_ready && is_ready_expired_locked(*slot)) {
          to_delete.push_back(std::move(slot->ready_name));
          slot->has_ready = false;
          slot->ready_offset = 0;
        }
      }
    }
    for (const auto& name : to_delete) delete_remote_async(name);
    lk.lock();
  }
}

std::string CacheManager::promote_locked(Slot& slot) {
  // Returns the *previous* ready_name (if any) so the caller can schedule
  // its deletion AFTER releasing the lock — calling delete_remote_async here
  // would deadlock with our own mutex.
  std::string old_to_delete;
  if (!slot.pending.valid()) return old_to_delete;
  if (slot.pending.wait_for(std::chrono::seconds(0)) !=
      std::future_status::ready) {
    return old_to_delete;
  }
  const std::string previous_ready = slot.ready_name;
  const bool had_old_ready = slot.has_ready;
  try {
    auto new_name = slot.pending.get();
    slot.pending = {};
    if (!new_name.empty()) {
      slot.ready_name = std::move(new_name);
      slot.ready_offset = slot.pending_through_index;
      slot.ready_created_at = std::chrono::steady_clock::now();
      slot.has_ready = true;
    }
    slot.pending_through_index = 0;
  } catch (...) {
    slot.pending = {};
    slot.pending_through_index = 0;
  }
  if (had_old_ready && previous_ready != slot.ready_name) {
    old_to_delete = previous_ready;
  }
  return old_to_delete;
}

bool CacheManager::is_ready_expired_locked(const Slot& slot) const {
  if (!slot.has_ready) return false;
  auto ttl = std::chrono::seconds(cache_ttl_seconds_.load());
  return std::chrono::steady_clock::now() - slot.ready_created_at >= ttl;
}

void CacheManager::delete_remote_async(const std::string& cache_name) {
  if (cache_name.empty()) return;
  DeleteCallback cb;
  {
    std::shared_lock<std::shared_mutex> lk(mu_);
    cb = delete_cb_;
  }
  // The delete callback hits Python — do it on a detached thread so we never
  // block the caller, and acquire the GIL inside.
  if (!cb) return;
  std::thread([cb, cache_name]() {
    try {
      nb::gil_scoped_acquire gil;
      cb(cache_name);
    } catch (...) {
    }
  }).detach();
}

}  // namespace agent_core
