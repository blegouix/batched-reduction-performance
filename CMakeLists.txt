# SPDX-FileCopyrightText: 2025 Baptiste Legouix
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)
project(BatchedReductionPerformance VERSION 0.0.0 LANGUAGES CXX CUDA)

## Google Benchmark
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
option(
    BENCHMARK_ENABLE_INSTALL
    "Enable installation of benchmark. (Projects embedding benchmark may want to turn this OFF.)"
    OFF
)
add_subdirectory(vendor/benchmark)

# Our project

## The library itself

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(batched_reduction_performance_lib INTERFACE)
target_include_directories(
    batched_reduction_performance_lib
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

## The benchmarks executable

add_subdirectory(benchmarks)
