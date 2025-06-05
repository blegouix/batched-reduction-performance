// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <iostream>

#include <cuda/std/mdspan>

#pragma once

namespace wrapper {

static constexpr std::size_t stride =
    32; // Attempt to optimize benchmarks using layout_stride according to warp
        // size

namespace detail {

template <std::size_t M, std::size_t N, class Layout> struct MakeMDSpan {
  static cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>,
                           Layout>
  run(double *ptr) {
    return cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>,
                             Layout>(ptr);
  }
};

template <std::size_t M, std::size_t N>
struct MakeMDSpan<M, N, cuda::std::layout_stride> {
  static cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>,
                           cuda::std::layout_stride>
  run(double *ptr) {
    return cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>,
                             cuda::std::layout_stride>(
        ptr, cuda::std::layout_stride::mapping<
                 cuda::std::extents<std::size_t, M, N>>{
                 cuda::std::extents<std::size_t, M, N>{},
                 cuda::std::array<std::size_t, 2>{stride, 1}});
  }
};

} // namespace detail

template <std::size_t M, std::size_t N, class Layout>
cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>,
                  cuda::std::layout_stride>
wrap(double *ptr) {
  return detail::MakeMDSpan<M, N, Layout>::run(ptr);
}

} // namespace wrapper
