# StarryVector top-level build script.
#
# Language policy: strict C++11 everywhere.  CI (and local builds) must
# never need a newer standard; please keep it that way (see
# notes/01-项目组成.md, "C++11 检查单").
cmake_minimum_required(VERSION 3.10)
project(StarryVector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++11, not gnu++11

if(NOT MSVC)
  add_compile_options(-Wall -Wextra -pedantic)
endif()

# Keep every produced binary under build/bin for easy scripting.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library: zero third-party dependencies by design.
# ---------------------------------------------------------------------------
add_library(starry_core STATIC
  src/types.cpp
  src/distance.cpp
  src/flat_index.cpp
  src/db.cpp)
target_include_directories(starry_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Benchmark driver (used by validation/run_validation.py).
# ---------------------------------------------------------------------------
add_executable(starry_bench apps/bench_main.cpp)
target_link_libraries(starry_bench PRIVATE starry_core Threads::Threads)

# ---------------------------------------------------------------------------
# Unit tests (doctest, vendored under third_party/, C++11 compatible).
# ---------------------------------------------------------------------------
enable_testing()
add_executable(starry_tests tests/test_core.cpp)
target_include_directories(starry_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(starry_tests PRIVATE starry_core Threads::Threads)
add_test(NAME core COMMAND starry_tests)
