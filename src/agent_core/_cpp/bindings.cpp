// Native module entry point for agent_core.
//
// Each subsystem (history store, SQLite writer, registry, cache manager,
// compaction engine) registers its bindings via a free function declared
// below. This file assembles them into the final Python module so each
// subsystem's binding code can live next to its implementation.

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

#include "agent_core/cache_manager.hpp"
#include "agent_core/history_store.hpp"
#include "agent_core/message_slot.hpp"
#include "agent_core/registry.hpp"
#include "agent_core/sqlite_writer.hpp"

namespace nb = nanobind;

namespace agent_core {

// Forward declarations for binding registrars defined in sibling files.
void register_session_bindings(nb::module_& m);
void register_cache_bindings(nb::module_& m);
void register_compaction_bindings(nb::module_& m);

namespace {

nb::str hello() { return nb::str("ok"); }

// Helper exposed to Python to flush every pending write — used in tests and
// at end-of-run durability points.
void flush_writes() {
  nb::gil_scoped_release release;
  SqliteWriter::instance().flush_all();
}

// Read-only access used by SqliteConversationStore's shim path.
std::vector<std::string> sqlite_load(const std::string& db_path,
                                     const std::string& session_id,
                                     const std::string& agent_type) {
  nb::gil_scoped_release release;
  return SqliteWriter::instance().load(db_path, session_id, agent_type);
}

// Module-level shutdown — Python atexit calls this so background threads and
// SQLite connections are torn down while the interpreter is still healthy
// (i.e. before static-destruction order becomes unpredictable).
void shutdown_native() {
  // The Registry's reaper holds no Python references; tearing it down first
  // is safe and stops any further activity that might call back into Python.
  // The cache manager has Python callbacks though — close it first so the
  // workers stop before Python disappears underneath them.
  CacheManager::instance().close();
  Registry::instance().shutdown();
  SqliteWriter::instance().shutdown();
}

void register_history_bindings(nb::module_& m) {
  nb::class_<HistoryStore>(m, "HistoryStore")
      .def(nb::init<std::string, std::string, std::string>(),
           nb::arg("session_id"), nb::arg("agent_type"), nb::arg("db_path") = "",
           "Open or create a HistoryStore for (session_id, agent_type). When "
           "db_path is non-empty the store persists through the background "
           "SQLite writer and rehydrates from disk on construction.")
      .def(
          "append",
          [](HistoryStore& self, nb::object native, std::string canonical_json,
             std::uint32_t approx_tokens, std::string role,
             std::string provider_tag) {
            self.append(MessageSlot(std::move(native), std::move(canonical_json),
                                    std::move(role), std::move(provider_tag),
                                    approx_tokens));
          },
          nb::arg("provider_native"), nb::arg("canonical_json"),
          nb::arg("approx_tokens"), nb::arg("role"), nb::arg("provider_tag"),
          "Append one already-canonicalized message. Releases the GIL while "
          "persisting; holds it while moving the Python ref into the slot.")
      .def(
          "replace_prefix",
          [](HistoryStore& self, std::size_t prefix_len, nb::object summary_native,
             std::string canonical_json, std::uint32_t approx_tokens,
             std::string role, std::string provider_tag) {
            self.replace_prefix(prefix_len,
                                MessageSlot(std::move(summary_native),
                                            std::move(canonical_json),
                                            std::move(role),
                                            std::move(provider_tag),
                                            approx_tokens));
          },
          nb::arg("prefix_len"), nb::arg("summary_native"),
          nb::arg("canonical_json"), nb::arg("approx_tokens"), nb::arg("role"),
          nb::arg("provider_tag"),
          "Replace the first prefix_len slots with a single summary message.")
      .def("clear", &HistoryStore::clear,
           "Drop all messages and enqueue a clear of the persisted row.")
      .def(
          "rehydrate_canonical",
          [](HistoryStore& self,
             std::vector<std::tuple<nb::object, std::string, std::uint32_t,
                                    std::string, std::string>>
                 slots) {
            std::vector<MessageSlot> built;
            built.reserve(slots.size());
            for (auto& tup : slots) {
              built.emplace_back(std::move(std::get<0>(tup)),
                                 std::move(std::get<1>(tup)),
                                 std::move(std::get<3>(tup)),
                                 std::move(std::get<4>(tup)), std::get<2>(tup));
            }
            self.rehydrate(std::move(built));
          },
          nb::arg("slots"),
          "Replace the entire store with the supplied slots — called after "
          "the Python wrapper re-attaches provider_native refs to canonical "
          "JSON loaded from disk.")
      .def(
          "snapshot_native",
          [](const HistoryStore& self) { return self.snapshot_native(); },
          "Return a list of provider_native references in order.")
      .def("snapshot_canonical", &HistoryStore::snapshot_canonical,
           "Return a list of canonical_json strings in order.")
      .def("snapshot_role_and_json", &HistoryStore::snapshot_role_and_json,
           "Return (role, canonical_json) pairs in order — used for compaction "
           "tail selection without materializing native refs.")
      .def("get_native", &HistoryStore::get_native, nb::arg("index"),
           "Return the provider_native at index, or raise IndexError.")
      .def("__len__", &HistoryStore::size)
      .def("size", &HistoryStore::size)
      .def("total_approx_tokens", &HistoryStore::total_approx_tokens,
           "Sum of cached approx_tokens across the store.")
      .def("flush", &HistoryStore::flush,
           "Block until any pending persistence is written.")
      .def_prop_ro(
          "session_id",
          [](const HistoryStore& self) { return self.session_id(); })
      .def_prop_ro(
          "agent_type",
          [](const HistoryStore& self) { return self.agent_type(); })
      .def_prop_ro("db_path",
                   [](const HistoryStore& self) { return self.db_path(); })
      .def_prop_ro("is_persistent", &HistoryStore::is_persistent);
}

}  // namespace
}  // namespace agent_core

NB_MODULE(_native, m) {
  m.doc() = "agent_core native extension (registry, history, cache, compaction)";
  m.attr("__version__") = "0.2.0";
  m.def("hello", &agent_core::hello,
        "Smoke-test entry point — returns 'ok' if the extension loaded.");
  m.def("flush_writes", &agent_core::flush_writes,
        "Block until every queued SQLite write has been persisted.");
  m.def("sqlite_load", &agent_core::sqlite_load, nb::arg("db_path"),
        nb::arg("session_id"), nb::arg("agent_type"),
        "Read persisted canonical JSON strings for a session — used by the "
        "Python conversation-store shim.");
  m.def("_shutdown", &agent_core::shutdown_native,
        "Tear down background threads and SQLite connections. Registered as "
        "an atexit handler by the Python wrapper so cleanup runs before "
        "static-destruction order takes over.");

  agent_core::register_history_bindings(m);
  agent_core::register_session_bindings(m);
  agent_core::register_cache_bindings(m);
  agent_core::register_compaction_bindings(m);
}
