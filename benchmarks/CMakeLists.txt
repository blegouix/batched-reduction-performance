
# SPDX-FileCopyrightText: 2025 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_executable(batched-reduction-performance benchmark.cpp)
target_link_libraries(batched-reduction-performance PUBLIC benchmark::benchmark batched_reduction_performance_lib)
