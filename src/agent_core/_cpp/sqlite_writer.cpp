#include "agent_core/sqlite_writer.hpp"

#include <sqlite3.h>

#include <chrono>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace agent_core {

namespace {

// JSON-encode a vector of already-canonical strings as a JSON array. Each
// element is already valid JSON, so this is just bracketing + commas. We avoid
// pulling in a full JSON library here: the input shape is fixed.
std::string encode_history_array(const std::vector<std::string>& canonical) {
  if (canonical.empty()) return "[]";
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < canonical.size(); ++i) {
    if (i != 0) out << ',';
    out << canonical[i];
  }
  out << ']';
  return out.str();
}

std::string make_key(const std::string& db,
                     const std::string& session,
                     const std::string& agent) {
  std::string key;
  key.reserve(db.size() + session.size() + agent.size() + 2);
  key.append(db).push_back('\x1f');
  key.append(session).push_back('\x1f');
  key.append(agent);
  return key;
}

}  // namespace

SqliteWriter& SqliteWriter::instance() {
  static SqliteWriter inst;
  return inst;
}

SqliteWriter::SqliteWriter() : worker_(&SqliteWriter::worker_loop, this) {}

SqliteWriter::~SqliteWriter() { shutdown(); }

void SqliteWriter::submit(WriteRequest request) {
  std::string key = make_key(request.db_path, request.session_id, request.agent_type);
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto& slot = pending_[std::move(key)];
    slot.request = std::move(request);
    slot.dirty = true;
  }
  cv_.notify_one();
}

std::vector<std::string> SqliteWriter::load(const std::string& db_path,
                                            const std::string& session_id,
                                            const std::string& agent_type) {
  // Check the in-flight queue first; an unwritten pending state takes
  // precedence over the on-disk row.
  {
    std::string key = make_key(db_path, session_id, agent_type);
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(key);
    if (it != pending_.end()) {
      if (it->second.request.clear) {
        return {};
      }
      return it->second.request.canonical_messages;
    }
  }

  sqlite3* conn;
  {
    std::lock_guard<std::mutex> lk(mu_);
    conn = get_or_open(db_path);
  }
  if (!conn) return {};

  const char* sql =
      "SELECT history FROM conversations "
      "WHERE session_id = ? AND agent_type = ?";
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(conn, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    return {};
  }
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 2, agent_type.c_str(), -1, SQLITE_TRANSIENT);

  std::vector<std::string> result;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    const unsigned char* raw = sqlite3_column_text(stmt, 0);
    int len = sqlite3_column_bytes(stmt, 0);
    if (raw && len > 0) {
      // Parse the JSON array of canonical JSON strings. We don't ship a JSON
      // library; the array shape is `[{...},{...},...]` with no embedded
      // newlines at the top level (json.dumps without indent). We walk it
      // tracking brace depth so commas inside objects don't split.
      std::string body(reinterpret_cast<const char*>(raw),
                       static_cast<std::size_t>(len));
      if (body.size() >= 2 && body.front() == '[' && body.back() == ']') {
        int depth = 0;
        bool in_string = false;
        bool escape = false;
        std::size_t start = 1;
        for (std::size_t i = 1; i + 1 < body.size(); ++i) {
          char c = body[i];
          if (in_string) {
            if (escape) {
              escape = false;
            } else if (c == '\\') {
              escape = true;
            } else if (c == '"') {
              in_string = false;
            }
            continue;
          }
          if (c == '"') {
            in_string = true;
          } else if (c == '{' || c == '[') {
            ++depth;
          } else if (c == '}' || c == ']') {
            --depth;
          } else if (c == ',' && depth == 0) {
            // Trim leading whitespace from each element so we keep the
            // canonical JSON exactly as the producer wrote it.
            std::size_t s = start;
            while (s < i && (body[s] == ' ' || body[s] == '\n' || body[s] == '\t')) {
              ++s;
            }
            result.emplace_back(body.substr(s, i - s));
            start = i + 1;
          }
        }
        // Final element.
        std::size_t s = start;
        std::size_t e = body.size() - 1;
        while (s < e && (body[s] == ' ' || body[s] == '\n' || body[s] == '\t')) {
          ++s;
        }
        if (s < e) {
          result.emplace_back(body.substr(s, e - s));
        }
      }
    }
  }
  sqlite3_finalize(stmt);
  return result;
}

void SqliteWriter::flush_all() {
  std::unique_lock<std::mutex> lk(mu_);
  // Drain every dirty request inline on the calling thread. The worker
  // thread may also pick them up; we hold the lock while draining so we and
  // the worker don't race on the same key.
  while (true) {
    std::string key;
    Coalesced* slot = nullptr;
    for (auto& [k, c] : pending_) {
      if (c.dirty) {
        key = k;
        slot = &c;
        break;
      }
    }
    if (!slot) break;
    WriteRequest req = slot->request;
    slot->dirty = false;
    lk.unlock();
    execute_request(req);
    lk.lock();
  }
}

void SqliteWriter::shutdown() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) return;
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();

  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [path, conn] : connections_) {
    if (conn) sqlite3_close_v2(conn);
  }
  connections_.clear();
}

void SqliteWriter::worker_loop() {
  std::unique_lock<std::mutex> lk(mu_);
  while (!stop_) {
    // Wait for either: a new write submission, or shutdown.
    cv_.wait_for(lk, kDebounce, [&] { return stop_ || !pending_.empty(); });
    if (stop_) break;

    // After the wait, allow up to one more debounce window for additional
    // mutations on the same key to coalesce.
    if (!pending_.empty()) {
      cv_.wait_for(lk, kDebounce, [&] { return stop_; });
      if (stop_) break;
    }

    // Snapshot current dirty state, mark clean, drop the lock, then write.
    std::vector<WriteRequest> batch;
    batch.reserve(pending_.size());
    for (auto& [key, slot] : pending_) {
      if (slot.dirty) {
        batch.push_back(slot.request);
        slot.dirty = false;
      }
    }
    if (batch.empty()) continue;
    lk.unlock();
    for (const auto& req : batch) {
      execute_request(req);
    }
    lk.lock();
  }
}

sqlite3* SqliteWriter::get_or_open(const std::string& db_path) {
  auto it = connections_.find(db_path);
  if (it != connections_.end()) {
    return it->second;
  }
  sqlite3* conn = nullptr;
  int rc = sqlite3_open_v2(
      db_path.c_str(), &conn,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
      nullptr);
  if (rc != SQLITE_OK) {
    if (conn) sqlite3_close_v2(conn);
    return nullptr;
  }
  // WAL mode + normal sync is the right durability/perf trade for a writer
  // that already coalesces on its own. fsync per commit becomes optional.
  sqlite3_exec(conn, "PRAGMA journal_mode = WAL;", nullptr, nullptr, nullptr);
  sqlite3_exec(conn, "PRAGMA synchronous = NORMAL;", nullptr, nullptr, nullptr);
  sqlite3_exec(
      conn,
      "CREATE TABLE IF NOT EXISTS conversations ("
      "  session_id TEXT NOT NULL,"
      "  agent_type TEXT NOT NULL,"
      "  history    TEXT NOT NULL,"
      "  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
      "  PRIMARY KEY (session_id, agent_type)"
      ");",
      nullptr, nullptr, nullptr);
  connections_.emplace(db_path, conn);
  return conn;
}

void SqliteWriter::execute_request(const WriteRequest& req) {
  sqlite3* conn = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    conn = get_or_open(req.db_path);
  }
  if (!conn) return;

  if (req.clear) {
    const char* sql =
        "DELETE FROM conversations WHERE session_id = ? AND agent_type = ?";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(conn, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
    sqlite3_bind_text(stmt, 1, req.session_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, req.agent_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return;
  }

  const std::string body = encode_history_array(req.canonical_messages);
  const char* sql =
      "INSERT INTO conversations (session_id, agent_type, history, updated_at) "
      "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
      "ON CONFLICT (session_id, agent_type) "
      "DO UPDATE SET history = excluded.history, updated_at = CURRENT_TIMESTAMP";
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(conn, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
  sqlite3_bind_text(stmt, 1, req.session_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 2, req.agent_type.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 3, body.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

std::string SqliteWriter::encode_history(
    const std::vector<std::string>& canonical) {
  return encode_history_array(canonical);
}

}  // namespace agent_core
