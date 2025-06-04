# SPDX-FileCopyrightText: 2025 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_library("printer" INTERFACE)

target_include_directories("printer" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
