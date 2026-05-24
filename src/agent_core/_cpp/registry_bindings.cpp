// Python bindings for Registry, Session, and the SessionHandle RAII type.
//
// The SessionHandle exposed to Python is a refcounted wrapper around
// std::shared_ptr<Session>. Dropping it (via __del__ or explicit close())
// runs Registry::release() exactly once.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "agent_core/history_store.hpp"
#include "agent_core/registry.hpp"
#include "agent_core/session.hpp"

namespace nb = nanobind;

namespace agent_core {

// Python-side wrapper exposing only the safe surface of a Session and managing
// release-on-drop. Holding a SessionHandle keeps the underlying Session alive
// (via shared_ptr) and contributes one ref to the registry's refcount.
class SessionHandle {
 public:
  SessionHandle(std::shared_ptr<Session> session, bool counted)
      : session_(std::move(session)), counted_(counted) {}

  ~SessionHandle() { close(); }

  SessionHandle(const SessionHandle&) = delete;
  SessionHandle& operator=(const SessionHandle&) = delete;

  void close() {
    if (closed_) return;
    closed_ = true;
    if (counted_ && session_) {
      Registry::instance().release(session_->session_id());
    }
    session_.reset();
  }

  std::string session_id() const { return session_->session_id(); }
  std::string agent_type() const { return session_->agent_type(); }
  std::string db_path() const { return session_->db_path(); }

  bool is_cancelled() const { return session_ && session_->is_cancelled(); }

  // Per-session cancel — without touching the subtree. Use the registry's
  // cancel_subtree to propagate.
  void cancel() {
    if (session_) session_->cancel();
  }

  void clear_cancellation() {
    if (session_) session_->clear_cancellation();
  }

  std::shared_ptr<HistoryStore> history() const {
    return session_->history_ptr();
  }

  std::uint32_t ref_count() const {
    return session_ ? session_->ref_count() : 0u;
  }

 private:
  std::shared_ptr<Session> session_;
  bool counted_;
  bool closed_ = false;
};

namespace {

// Python-facing helpers on the registry singleton.
std::unique_ptr<SessionHandle> acquire_handle(const std::string& session_id,
                                              const std::string& agent_type,
                                              const std::string& db_path) {
  SessionConfig cfg{session_id, agent_type, db_path};
  auto session = Registry::instance().acquire(std::move(cfg));
  return std::make_unique<SessionHandle>(std::move(session), /*counted=*/true);
}

std::unique_ptr<SessionHandle> peek_handle(const std::string& session_id) {
  auto session = Registry::instance().get(session_id);
  if (!session) return nullptr;
  // Peeking does NOT bump the refcount — the handle owns no release.
  return std::make_unique<SessionHandle>(std::move(session), /*counted=*/false);
}

}  // namespace

void register_session_bindings(nb::module_& m) {
  nb::class_<SessionHandle>(m, "SessionHandle")
      .def("close", &SessionHandle::close,
           "Drop this handle's reference to the underlying session. Idempotent.")
      .def("cancel", &SessionHandle::cancel,
           "Set the session's cancellation flag (does not touch descendants — "
           "use registry.cancel_subtree for that).")
      .def("clear_cancellation", &SessionHandle::clear_cancellation,
           "Clear this session's cancellation flag.")
      .def("history", &SessionHandle::history,
           "Return the HistoryStore for this session.")
      .def("__enter__", [](nb::handle self) -> nb::handle { return self; })
      .def(
          "__exit__",
          [](SessionHandle& self, nb::handle, nb::handle, nb::handle) {
            self.close();
            return false;
          },
          nb::arg("exc_type").none(), nb::arg("exc_value").none(),
          nb::arg("traceback").none())
      .def_prop_ro("session_id", &SessionHandle::session_id)
      .def_prop_ro("agent_type", &SessionHandle::agent_type)
      .def_prop_ro("db_path", &SessionHandle::db_path)
      .def_prop_ro("is_cancelled", &SessionHandle::is_cancelled)
      .def_prop_ro("ref_count", &SessionHandle::ref_count);

  // Wrap the singleton as a stateless module-level helper class so Python
  // sees a clean object surface (``_native.registry.acquire(...)``).
  struct RegistryProxy {};
  nb::class_<RegistryProxy>(m, "_RegistryProxy")
      .def_static("acquire", &acquire_handle, nb::arg("session_id"),
                  nb::arg("agent_type"), nb::arg("db_path") = "",
                  "Acquire (or resurrect) a session. Bumps refcount; the "
                  "returned SessionHandle releases on close/__del__.")
      .def_static("peek", &peek_handle, nb::arg("session_id"),
                  "Return a non-counted handle if the session is resident.")
      .def_static(
          "cancel_subtree",
          [](const std::string& root) {
            return Registry::instance().cancel_subtree(root);
          },
          nb::arg("root_session_id"),
          "Set the cancellation flag on root and every descendant. Returns "
          "the number of sessions flagged.")
      .def_static(
          "clear_cancellation",
          [](const std::string& session_id, bool recursive) {
            Registry::instance().clear_cancellation(session_id, recursive);
          },
          nb::arg("session_id"), nb::arg("recursive") = false,
          "Clear the cancellation flag. With recursive=True, also clears "
          "every descendant.")
      .def_static(
          "list_active",
          [] { return Registry::instance().list_active(); },
          "List every resident session id in sorted order.")
      .def_static(
          "descendants",
          [](const std::string& root, bool include_root) {
            return Registry::instance().descendants(root, include_root);
          },
          nb::arg("root_session_id"), nb::arg("include_root") = false,
          "List session ids under root (sorted).")
      .def_static(
          "set_idle_ttl_seconds",
          [](std::uint32_t s) { Registry::instance().set_idle_ttl_seconds(s); },
          nb::arg("seconds"),
          "Configure how long an unreferenced session stays resident before "
          "the reaper evicts it.")
      .def_static(
          "idle_ttl_seconds",
          [] { return Registry::instance().idle_ttl_seconds(); })
      .def_static(
          "reap_now", [] { return Registry::instance().reap_now(); },
          "Synchronously reap idle sessions. Returns the number evicted.")
      .def_static(
          "clear", [] { Registry::instance().clear(); },
          "Test-only: evict every resident session (does not touch SQLite).");

  m.attr("registry") = RegistryProxy{};
}

}  // namespace agent_core
