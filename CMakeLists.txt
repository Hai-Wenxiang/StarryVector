# StarryVector top-level build script.
#
# Language policy: C++17 everywhere.
cmake_minimum_required(VERSION 3.10)
project(StarryVector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++17, not gnu++17

add_compile_options(-Wall -Wextra -pedantic)

# Keep every produced binary under build/bin for easy scripting.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library: zero third-party dependencies by design.
#
# distance_avx2.cpp is compiled with -mavx2 -mfma on x86 only; the kernels
# are gated at runtime by __builtin_cpu_supports (see src/distance.cpp), so
# the resulting binary still runs on CPUs without AVX2.
# ---------------------------------------------------------------------------
add_library(starry_core STATIC
  src/types.cpp
  src/distance.cpp
  src/flat_index.cpp
  src/db.cpp)

option(STARRY_DISABLE_SIMD "Do not build SIMD kernels (scalar only)" OFF)
if(NOT STARRY_DISABLE_SIMD AND (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)"))
  target_sources(starry_core PRIVATE src/distance_avx2.cpp)
  set_source_files_properties(src/distance_avx2.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma")
  target_compile_definitions(starry_core PRIVATE STARRY_HAVE_AVX2=1)
endif()
target_include_directories(starry_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Benchmark driver (used by validation/run_validation.py).
# ---------------------------------------------------------------------------
add_executable(starry_bench apps/bench_main.cpp)
target_link_libraries(starry_bench PRIVATE starry_core Threads::Threads)

# ---------------------------------------------------------------------------
# RAG retrieval example (see examples/rag_demo.cpp).
# ---------------------------------------------------------------------------
add_executable(starry_rag_demo examples/rag_demo.cpp)
target_link_libraries(starry_rag_demo PRIVATE starry_core Threads::Threads)

# ---------------------------------------------------------------------------
# Unit tests (doctest, vendored under third_party/, C++11 compatible).
# ---------------------------------------------------------------------------
enable_testing()
add_executable(starry_tests tests/test_core.cpp)
target_include_directories(starry_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(starry_tests PRIVATE starry_core Threads::Threads)
add_test(NAME core COMMAND starry_tests)
