# StarryVector top-level build script.
#
# Language policy: C++17 everywhere.
cmake_minimum_required(VERSION 3.10)
project(StarryVector VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++17, not gnu++17

add_compile_options(-Wall -Wextra -pedantic)

# Keep every produced binary under build/bin for easy scripting.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Install destinations follow the GNU conventions (lib, include, ...).
include(GNUInstallDirs)

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
  src/hnsw_index.cpp
  src/wal.cpp
  src/db.cpp)

option(STARRY_DISABLE_SIMD "Do not build SIMD kernels (scalar only)" OFF)
if(NOT STARRY_DISABLE_SIMD AND (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)"))
  target_sources(starry_core PRIVATE src/distance_avx2.cpp)
  set_source_files_properties(src/distance_avx2.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma")
  target_compile_definitions(starry_core PRIVATE STARRY_HAVE_AVX2=1)
endif()
target_include_directories(starry_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

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

# Storage-engine fault-injection suite: run standalone or with scenario
# numbers (`starry_crash_suite 3 15`); see the header comment inside.
add_executable(starry_crash_suite tests/crash_suite.cpp)
target_link_libraries(starry_crash_suite PRIVATE starry_core Threads::Threads)
add_test(NAME crash_suite COMMAND starry_crash_suite)

# ---------------------------------------------------------------------------
# Install rules + package export, so downstream projects can consume
# StarryVector either via `cmake --install` + find_package(StarryVector) or
# via add_subdirectory().  The library has no public dependencies, so the
# package config is just a shim that includes the exported targets.
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS starry_core
  EXPORT StarryVectorTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/starry/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/starry)

install(EXPORT StarryVectorTargets
  FILE StarryVectorTargets.cmake
  NAMESPACE StarryVector::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StarryVector)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/StarryVectorConfig.cmake
  "include(\"\${CMAKE_CURRENT_LIST_DIR}/StarryVectorTargets.cmake\")\n")

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/StarryVectorConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/StarryVectorConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/StarryVectorConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StarryVector)
