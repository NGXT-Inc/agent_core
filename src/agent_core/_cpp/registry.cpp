#include "agent_core/registry.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace agent_core {

namespace {

constexpr char kSeparator = ':';

}  // namespace

Registry& Registry::instance() {
  static Registry inst;
  return inst;
}

Registry::Registry() : reaper_thread_(&Registry::reaper_loop, this) {}

Registry::~Registry() { shutdown(); }

std::shared_ptr<Session> Registry::acquire(SessionConfig cfg) {
  // Try the fast path under a shared lock first.
  {
    std::shared_lock<std::shared_mutex> lk(mu_);
    auto it = sessions_.find(cfg.session_id);
    if (it != sessions_.end()) {
      it->second->inc_ref();
      it->second->touch();
      return it->second;
    }
  }

  // Construct a new Session — releasing the GIL is fine here since we hold
  // no Python state. The HistoryStore constructor does its own GIL juggling
  // around SQLite I/O.
  auto history = std::make_shared<HistoryStore>(cfg.session_id, cfg.agent_type,
                                                cfg.db_path);

  std::unique_lock<std::shared_mutex> lk(mu_);
  // Double-check under the unique lock: another thread may have created the
  // session between our shared-lock check and now.
  auto it = sessions_.find(cfg.session_id);
  if (it != sessions_.end()) {
    it->second->inc_ref();
    it->second->touch();
    return it->second;
  }
  auto session = std::make_shared<Session>(std::move(cfg), std::move(history));
  session->inc_ref();
  session->touch();
  auto sid = session->session_id();
  sessions_.emplace(std::move(sid), session);
  return session;
}

std::shared_ptr<Session> Registry::get(const std::string& session_id) const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  auto it = sessions_.find(session_id);
  if (it == sessions_.end()) return nullptr;
  return it->second;
}

void Registry::release(const std::string& session_id) {
  std::shared_lock<std::shared_mutex> lk(mu_);
  auto it = sessions_.find(session_id);
  if (it == sessions_.end()) return;
  it->second->dec_ref();
  it->second->touch();
}

std::size_t Registry::cancel_subtree(const std::string& root) {
  std::size_t count = 0;
  std::shared_lock<std::shared_mutex> lk(mu_);
  for (auto& [sid, session] : sessions_) {
    if (is_descendant(sid, root) || sid == root) {
      session->cancel();
      ++count;
    }
  }
  return count;
}

void Registry::clear_cancellation(const std::string& session_id, bool recursive) {
  std::shared_lock<std::shared_mutex> lk(mu_);
  if (!recursive) {
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) it->second->clear_cancellation();
    return;
  }
  for (auto& [sid, session] : sessions_) {
    if (is_descendant(sid, session_id) || sid == session_id) {
      session->clear_cancellation();
    }
  }
}

std::vector<std::string> Registry::list_active() const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  std::vector<std::string> out;
  out.reserve(sessions_.size());
  for (const auto& [sid, _] : sessions_) out.push_back(sid);
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> Registry::descendants(const std::string& root,
                                               bool include_root) const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  std::vector<std::string> out;
  for (const auto& [sid, _] : sessions_) {
    if (sid == root) {
      if (include_root) out.push_back(sid);
    } else if (is_descendant(sid, root)) {
      out.push_back(sid);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

void Registry::set_idle_ttl_seconds(std::uint32_t seconds) {
  idle_ttl_seconds_.store(seconds, std::memory_order_release);
}

std::uint32_t Registry::idle_ttl_seconds() const {
  return idle_ttl_seconds_.load(std::memory_order_acquire);
}

std::size_t Registry::reap_now() {
  const auto ttl_ns =
      static_cast<std::uint64_t>(idle_ttl_seconds_.load()) * 1'000'000'000ULL;
  const auto now_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());

  std::vector<std::string> evict;
  {
    std::shared_lock<std::shared_mutex> lk(mu_);
    for (const auto& [sid, session] : sessions_) {
      if (session->ref_count() == 0 &&
          now_ns - session->last_access_ns() >= ttl_ns) {
        evict.push_back(sid);
      }
    }
  }
  if (evict.empty()) return 0;

  std::unique_lock<std::shared_mutex> lk(mu_);
  std::size_t reaped = 0;
  for (const auto& sid : evict) {
    auto it = sessions_.find(sid);
    if (it == sessions_.end()) continue;
    // Recheck under the unique lock — another thread may have re-acquired.
    if (it->second->ref_count() == 0 &&
        now_ns - it->second->last_access_ns() >= ttl_ns) {
      sessions_.erase(it);
      ++reaped;
    }
  }
  return reaped;
}

void Registry::clear() {
  std::unique_lock<std::shared_mutex> lk(mu_);
  sessions_.clear();
}

void Registry::shutdown() {
  {
    std::lock_guard<std::mutex> lk(reaper_mu_);
    if (reaper_stop_) return;
    reaper_stop_ = true;
  }
  reaper_cv_.notify_all();
  if (reaper_thread_.joinable()) reaper_thread_.join();
}

void Registry::reaper_loop() {
  std::unique_lock<std::mutex> lk(reaper_mu_);
  // Wake every 60s to sweep idle sessions; an early notify_all on shutdown
  // breaks us out without waiting for the timer.
  while (!reaper_stop_) {
    reaper_cv_.wait_for(lk, std::chrono::seconds(60), [&] { return reaper_stop_; });
    if (reaper_stop_) break;
    lk.unlock();
    reap_now();
    lk.lock();
  }
}

bool Registry::is_descendant(const std::string& sid, const std::string& root) {
  if (sid.size() <= root.size()) return false;
  if (sid.compare(0, root.size(), root) != 0) return false;
  return sid[root.size()] == kSeparator;
}

}  // namespace agent_core
