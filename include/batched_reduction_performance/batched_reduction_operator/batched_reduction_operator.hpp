// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cooperative_groups/reduce.h>
#include <cuda/std/mdspan>

#pragma once

namespace batched_reduction_operator {

// Sequential operator

namespace detail {

template <std::size_t M, std::size_t N>
__global__ void sequential_kernel(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> data_out,
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>> data_in) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

#if defined ALLOW_UNCOMPLETE_WARP
  if (i < M) {
#else
  {
    static_assert(M % 32 == 0, "Uncomplete warps are not allowed, fix the "
                               "problem sizes or enable ALLOW_UNCOMPLETE_WARP");
#endif
    double sum = 0;

    for (std::size_t j = 0; j < N; ++j) {
      sum += data_in(i, j);
    }

    data_out(i) = sum;
  }
}

} // namespace detail

template <std::size_t BlockDim> class Sequential {
public:
  template <std::size_t M, std::size_t N>
  static void
  run(cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> data_out,
      cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>>
          data_in) {
    dim3 const blockDim(BlockDim);
    dim3 const gridDim((M + blockDim.x - 1) / blockDim.x);

    detail::sequential_kernel<<<gridDim, blockDim>>>(data_out, data_in);
  }
};

// Parallel operator using CUDA cooperative groups

namespace detail {

template <std::size_t M, std::size_t N>
__global__ void cooperative_groups_kernel(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> data_out,
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>> data_in) {
  /*
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  */

  std::size_t i = blockIdx.x;
  std::size_t j = threadIdx.x;

  double val;
#if defined ALLOW_UNCOMPLETE_WARP
  if (i < M && j < N) {
#else
  {
    static_assert(M % 32 == 0 && N % 32 == 0,
                  "Uncomplete warps are not allowed, fix the problem sizes or "
                  "enable ALLOW_UNCOMPLETE_WARP");
#endif
    val = data_in(i, j);
  }
#if defined ALLOW_UNCOMPLETE_WARP
  else {
#else
  {
    static_assert(M % 32 == 0 && N % 32 == 0,
                  "Uncomplete warps are not allowed, fix the problem sizes or "
                  "enable ALLOW_UNCOMPLETE_WARP");
#endif
    val = 0.;
  }

  // Perform reduction within the block
  double sum =
      cooperative_groups::reduce(cooperative_groups::this_thread_block(), val,
                                 cooperative_groups::plus<double>());

  // Thread 0 writes result to output
  if (cooperative_groups::this_thread_block().thread_rank() == 0) {
    data_out(i) = sum;
  }
}

} // namespace detail

template <std::size_t BlockDim> class CooperativeGroups {
public:
  template <std::size_t M, std::size_t N>
  static void
  run(cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> data_out,
      cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>>
          data_in) {
    dim3 const blockDim(BlockDim);
    dim3 const gridDim((M + blockDim.x - 1) / blockDim.x);

    detail::cooperative_groups_kernel<<<gridDim, blockDim>>>(data_out, data_in);
  }
};

} // namespace batched_reduction_operator
