# SPDX-FileCopyrightText: 2025 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_library("batched_reduction_kernel" INTERFACE)

target_include_directories("batched_reduction_kernel" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
