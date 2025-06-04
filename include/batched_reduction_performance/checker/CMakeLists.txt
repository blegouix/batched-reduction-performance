# SPDX-FileCopyrightText: 2025 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_library("checker" INTERFACE)

target_include_directories("checker" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
