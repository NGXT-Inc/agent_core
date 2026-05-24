// Process-wide background SQLite writer for HistoryStore persistence.
//
// Design goals:
//   * Single writer thread per process — one connection per database path is
//     opened lazily and reused, avoiding the open/commit/close cycle that the
//     pure-Python store paid on every mutation.
//   * Debounced coalescing — mutating the same (db_path, session_id) twice in
//     rapid succession results in one write, holding the LATEST snapshot. The
//     UPSERT is idempotent so a missed intermediate state is safe.
//   * Synchronous flush available — tests and shutdown call flush_all() to
//     guarantee durability before observation.
//   * GIL-free — all SQLite work happens with the GIL released; the writer
//     never calls back into Python.

#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct sqlite3;

namespace agent_core {

// One queued write request. ``canonical_messages`` is the *complete* current
// state for this session — the writer overwrites the row, it never appends.
struct WriteRequest {
  std::string db_path;
  std::string session_id;
  std::string agent_type;
  std::vector<std::string> canonical_messages;  // JSON strings, in order
  bool clear = false;                            // true → DELETE row instead of UPSERT
};

// Singleton background writer. The first ``instance()`` call spawns the worker
// thread; subsequent calls return the same reference. ``shutdown()`` is wired
// to module teardown via Python atexit.
class SqliteWriter {
 public:
  static SqliteWriter& instance();

  // Submit a save. The request is coalesced into the latest pending state for
  // (db_path, session_id, agent_type); only the newest snapshot is written.
  void submit(WriteRequest request);

  // Load a session's persisted history. Returns the canonical JSON strings in
  // insertion order, or an empty vector if the row does not exist. This is a
  // synchronous read that briefly serializes with the writer thread to grab
  // any in-flight pending state.
  std::vector<std::string> load(const std::string& db_path,
                                const std::string& session_id,
                                const std::string& agent_type);

  // Block until every queued write has been persisted. Used by tests and
  // optionally at end-of-run for durability points.
  void flush_all();

  // Stop the writer thread and close connections. Idempotent.
  void shutdown();

  SqliteWriter(const SqliteWriter&) = delete;
  SqliteWriter& operator=(const SqliteWriter&) = delete;

 private:
  SqliteWriter();
  ~SqliteWriter();

  void worker_loop();
  sqlite3* get_or_open(const std::string& db_path);
  void execute_request(const WriteRequest& req);
  static std::string encode_history(const std::vector<std::string>& canonical);

  struct Coalesced {
    WriteRequest request;
    bool dirty = true;
  };

  std::mutex mu_;
  std::condition_variable cv_;
  std::unordered_map<std::string, Coalesced> pending_;  // key: db_path|session|agent_type
  std::unordered_map<std::string, sqlite3*> connections_;
  bool stop_ = false;
  std::thread worker_;

  // Debounce window — how long to wait for additional coalescable writes
  // before flushing the dirty set. Small enough that durability lag is bounded;
  // large enough that a single tool-call round (≤4 mutations) collapses into
  // one write.
  static constexpr std::chrono::milliseconds kDebounce{25};
};

}  // namespace agent_core
