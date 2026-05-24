cmake_minimum_required(VERSION 3.20)
project(agent_core_native LANGUAGES CXX)

# C++20 across the board — we use std::shared_mutex, std::string_view,
# concepts in a few spots, and starts_with() on strings.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position-independent code for the shared library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warnings — treat the native extension as production code.
if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Locate the Python development headers via scikit-build-core's hints.
find_package(Python 3.11
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Locate nanobind via the package the build front-end installed.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# All native sources, bindings, and CMake details live under src/agent_core/_cpp.
add_subdirectory(src/agent_core/_cpp)
