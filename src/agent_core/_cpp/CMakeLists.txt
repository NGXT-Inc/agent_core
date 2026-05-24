# Native extension sources for agent_core.
#
# Each phase of the C++ migration adds its own sources here. The bindings.cpp
# file glues every subsystem into a single Python module (`agent_core._native`).

set(NATIVE_SOURCES
  bindings.cpp
  cache_manager.cpp
  cache_manager_bindings.cpp
  compaction.cpp
  compaction_bindings.cpp
  history_store.cpp
  registry.cpp
  registry_bindings.cpp
  sqlite_writer.cpp
)

nanobind_add_module(_native
  NB_STATIC
  STABLE_ABI
  ${NATIVE_SOURCES}
)

target_include_directories(_native PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# SQLite is a required dependency from Phase 2 onward.
find_package(SQLite3 REQUIRED)
target_link_libraries(_native PRIVATE SQLite::SQLite3)

# Install the extension next to the Python package.
install(TARGETS _native LIBRARY DESTINATION agent_core)
