// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <iostream>

#include <cuda/std/mdspan>

#pragma once

namespace wrapper {

static constexpr std::size_t stride =
    32; // Attempt to optimize benchmarks using layout_stride according to warp
        // size

enum LayoutEnum { Left, LeftStride, RightStride, Right };

namespace detail {

template <LayoutEnum Layout> struct MDSpanLayout;

template <> struct MDSpanLayout<Left> {
  using type = cuda::std::layout_left;
};

template <> struct MDSpanLayout<LeftStride> {
  using type = cuda::std::layout_stride;
};

template <> struct MDSpanLayout<RightStride> {
  using type = cuda::std::layout_stride;
};

template <> struct MDSpanLayout<Right> {
  using type = cuda::std::layout_right;
};

} // namespace detail

template <LayoutEnum Layout>
using mdspan_layout_t = typename detail::MDSpanLayout<Layout>::type;

namespace detail {

template <std::size_t M, std::size_t N, LayoutEnum Layout> struct MakeMDSpan {
  static cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>,
                           mdspan_layout_t<Layout>>
  run(double *ptr) {
    return cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>,
                             mdspan_layout_t<Layout>>(ptr);
  }
};

template <std::size_t M, std::size_t N> struct MakeMDSpan<M, N, LeftStride> {
  static cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>,
                           cuda::std::layout_stride>
  run(double *ptr) {
    return cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>,
                             cuda::std::layout_stride>(
        ptr, cuda::std::layout_stride::mapping<
                 cuda::std::extents<std::size_t, M, N>>{
                 cuda::std::extents<std::size_t, M, N>{},
                 cuda::std::array<std::size_t, 2>{1, stride}});
  }
};

template <std::size_t M, std::size_t N> struct MakeMDSpan<M, N, RightStride> {
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

template <std::size_t M, std::size_t N, LayoutEnum Layout>
cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>,
                  mdspan_layout_t<Layout>>
wrap(double *ptr) {
  return detail::MakeMDSpan<M, N, Layout>::run(ptr);
}

} // namespace wrapper
