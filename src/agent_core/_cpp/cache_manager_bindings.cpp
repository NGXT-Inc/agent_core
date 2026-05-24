// Python bindings for the C++ CacheManager.
//
// The public Python wrapper (``agent_core.core.caching.ContextCacheRegistry``)
// delegates to this binding. Vertex create/delete operations live in Python
// — the wrapper passes them in via ``configure``.

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>

#include "agent_core/cache_manager.hpp"

namespace nb = nanobind;

namespace agent_core {

void register_cache_bindings(nb::module_& m) {
  nb::class_<CacheAdvice>(m, "CacheAdvice")
      .def(nb::init<>())
      .def_ro("cache_name", &CacheAdvice::cache_name)
      .def_ro("contents_offset", &CacheAdvice::contents_offset);

  nb::class_<CacheSlotSnapshot>(m, "CacheSlotSnapshot")
      .def_ro("model_name", &CacheSlotSnapshot::model_name)
      .def_ro("min_token_threshold", &CacheSlotSnapshot::min_token_threshold)
      .def_ro("ready_name", &CacheSlotSnapshot::ready_name)
      .def_ro("ready_offset", &CacheSlotSnapshot::ready_offset)
      .def_ro("has_ready", &CacheSlotSnapshot::has_ready)
      .def_ro("ready_expired", &CacheSlotSnapshot::ready_expired)
      .def_ro("pending", &CacheSlotSnapshot::pending)
      .def_ro("pending_through_index", &CacheSlotSnapshot::pending_through_index)
      .def_ro("last_cache_token_count",
              &CacheSlotSnapshot::last_cache_token_count)
      .def_ro("config_fingerprint", &CacheSlotSnapshot::config_fingerprint);

  struct CacheManagerProxy {};
  nb::class_<CacheManagerProxy>(m, "_CacheManagerProxy")
      .def_static(
          "configure",
          [](CacheManager::CreateCallback create,
             CacheManager::DeleteCallback del, std::uint32_t max_workers,
             std::uint32_t cache_ttl_seconds) {
            CacheManager::instance().configure(std::move(create), std::move(del),
                                               max_workers, cache_ttl_seconds);
          },
          nb::arg("create_callback"), nb::arg("delete_callback"),
          nb::arg("max_workers") = 4, nb::arg("cache_ttl_seconds") = 600,
          "Register the Vertex create/delete callbacks and spin up workers.")
      .def_static("is_configured",
                  [] { return CacheManager::instance().is_configured(); })
      .def_static(
          "register_agent",
          [](std::string agent_id, std::string model_name,
             std::uint32_t min_token_threshold) {
            CacheManager::instance().register_agent(
                std::move(agent_id), std::move(model_name), min_token_threshold);
          },
          nb::arg("agent_id"), nb::arg("model_name"),
          nb::arg("min_token_threshold") = 32768)
      .def_static(
          "unregister_agent",
          [](const std::string& agent_id) {
            CacheManager::instance().unregister_agent(agent_id);
          },
          nb::arg("agent_id"))
      .def_static(
          "get_advice",
          [](const std::string& agent_id, const std::string& fingerprint,
             bool wait, double wait_timeout_seconds) {
            return CacheManager::instance().get_advice(agent_id, fingerprint,
                                                      wait, wait_timeout_seconds);
          },
          nb::arg("agent_id"), nb::arg("fingerprint"), nb::arg("wait") = false,
          nb::arg("wait_timeout_seconds") = 30.0)
      .def_static(
          "notify",
          [](const std::string& agent_id, const std::string& fingerprint,
             std::uint32_t token_count, std::size_t contents_size,
             nb::object payload) {
            CacheManager::instance().notify(agent_id, fingerprint, token_count,
                                            contents_size, std::move(payload));
          },
          nb::arg("agent_id"), nb::arg("fingerprint"), nb::arg("token_count"),
          nb::arg("contents_size"), nb::arg("payload"))
      .def_static("invalidate",
                  [](const std::string& agent_id) {
                    CacheManager::instance().invalidate(agent_id);
                  })
      .def_static(
          "peek_slot",
          [](const std::string& agent_id) {
            return CacheManager::instance().peek_slot(agent_id);
          },
          nb::arg("agent_id"))
      .def_static(
          "seed_slot",
          [](const std::string& agent_id, std::string ready_name,
             std::size_t ready_offset, bool ready_expired_now, bool pending_done,
             std::string pending_cache_name, std::size_t pending_through_index,
             std::uint32_t last_cache_token_count,
             std::string config_fingerprint) {
            CacheManager::instance().seed_slot(
                agent_id, std::move(ready_name), ready_offset,
                ready_expired_now, pending_done, std::move(pending_cache_name),
                pending_through_index, last_cache_token_count,
                std::move(config_fingerprint));
          },
          nb::arg("agent_id"), nb::arg("ready_name") = std::string(),
          nb::arg("ready_offset") = 0, nb::arg("ready_expired_now") = false,
          nb::arg("pending_done") = false,
          nb::arg("pending_cache_name") = std::string(),
          nb::arg("pending_through_index") = 0,
          nb::arg("last_cache_token_count") = 0,
          nb::arg("config_fingerprint") = std::string())
      .def_static("clear_slots",
                  [] { return CacheManager::instance().clear_slots(); },
                  "Drop every registered slot without tearing down workers — "
                  "the per-instance facade closes through this path.")
      .def_static("close", [] { CacheManager::instance().close(); });

  m.attr("cache_manager") = CacheManagerProxy{};
  m.def("compute_cache_fingerprint", &compute_fingerprint,
        nb::arg("system_instruction"), nb::arg("tool_names"),
        "Compute the same 16-hex-char SHA-256 fingerprint the Python "
        "ContextCacheRegistry has always used.");
}

}  // namespace agent_core
