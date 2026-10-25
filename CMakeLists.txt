cmake_minimum_required(VERSION 3.25)
project(tickforge VERSION 0.1.0 LANGUAGES CXX)

option(TICKFORGE_BUILD_TESTS "Build unit tests" ON)
option(TICKFORGE_BUILD_BENCH "Build Google Benchmark microbenchmarks" ON)
option(TICKFORGE_PYTHON "Build the pybind11 Python module" OFF)
option(TICKFORGE_CUDA "Build the CUDA batch-feature kernel" OFF)
set(TICKFORGE_SANITIZE "" CACHE STRING "Sanitizers to enable, e.g. 'address,undefined' or 'thread'")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the static core is linked into the Python module
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(TICKFORGE_SANITIZE)
  add_compile_options(-fsanitize=${TICKFORGE_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${TICKFORGE_SANITIZE})
endif()

include(FetchContent)

# ---- core library ------------------------------------------------------------
add_library(tickforge_core STATIC
  src/tick_store.cpp
  src/synth.cpp
  src/backtest.cpp
  src/thread_pool.cpp
  src/sweep.cpp
  src/batch_features.cpp)
add_library(tickforge::core ALIAS tickforge_core)
target_include_directories(tickforge_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(tickforge_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
find_package(Threads REQUIRED)
target_link_libraries(tickforge_core PUBLIC Threads::Threads)

if(TICKFORGE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
  endif()
  target_sources(tickforge_core PRIVATE src/batch_features.cu)
  target_compile_definitions(tickforge_core PUBLIC TICKFORGE_HAS_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(tickforge_core PUBLIC CUDA::cudart)
endif()

add_executable(tickforge_cli apps/tickforge_cli.cpp)
target_link_libraries(tickforge_cli PRIVATE tickforge::core)
target_compile_options(tickforge_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)

# ---- tests -------------------------------------------------------------------
if(TICKFORGE_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)

  add_executable(tickforge_tests
    tests/test_tick_store.cpp
    tests/test_memory.cpp
    tests/test_spsc_ring.cpp
    tests/test_features.cpp
    tests/test_backtest.cpp
    tests/test_sweep.cpp
    tests/test_batch_features.cpp)
  target_link_libraries(tickforge_tests PRIVATE tickforge::core GTest::gtest_main)
  gtest_discover_tests(tickforge_tests DISCOVERY_TIMEOUT 60)

  # Separate binary: it replaces global operator new to count allocations.
  add_executable(test_no_alloc tests/test_no_alloc.cpp)
  target_link_libraries(test_no_alloc PRIVATE tickforge::core GTest::gtest_main)
  gtest_discover_tests(test_no_alloc DISCOVERY_TIMEOUT 60)
endif()

# ---- benchmarks --------------------------------------------------------------
if(TICKFORGE_BUILD_BENCH)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
  add_executable(tickforge_bench bench/bench_main.cpp)
  target_link_libraries(tickforge_bench PRIVATE tickforge::core benchmark::benchmark)
endif()

# ---- Python ------------------------------------------------------------------
if(TICKFORGE_PYTHON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(tickforge python/bindings.cpp)
  target_link_libraries(tickforge PRIVATE tickforge::core)
endif()
