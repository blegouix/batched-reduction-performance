# SPDX-FileCopyrightText: 2025 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_library("wrapper" INTERFACE)

target_include_directories("wrapper" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
