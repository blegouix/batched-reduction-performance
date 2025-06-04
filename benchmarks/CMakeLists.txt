
# SPDX-FileCopyrightText: 2025 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_executable(batched-reduction-performance benchmark.cu)
target_link_libraries(batched-reduction-performance PUBLIC std::mdspan benchmark::benchmark batched_reduction_performance_lib) # TODO remove dependency to mdspan here
