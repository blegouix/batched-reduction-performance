// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cuda/std/mdspan>

#pragma once

namespace batched_reduction_operator {

// Sequential operator

namespace detail {

template <std::size_t M, std::size_t N>
static __global__ void sequential_kernel(
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
    double tmp = 0;

    for (std::size_t j = 0; j < N; ++j) {
      tmp += data_in(i, j);
    }

    data_out(i) = tmp;
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

// Sequential operator with shared memory usage

namespace detail {

template <std::size_t M, std::size_t N>
static __global__ void sequential_with_shared_memory_kernel(
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

    // Shared memory buffer storing one row of data_in
    __shared__ double buffer[N];
    for (std::size_t j = 0; j < N; ++j) {
      buffer[j] = data_in(i, j);
    }

    __syncthreads();

    double tmp = 0;

    for (std::size_t j = 0; j < N; ++j) {
      tmp += data_in(i, j);
    }

    data_out(i) = tmp;
  }
}

} // namespace detail

template <std::size_t BlockDim> class SequentialWithSharedMemory {
public:
  template <std::size_t M, std::size_t N>
  static void
  run(cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> data_out,
      cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>>
          data_in) {
    dim3 const blockDim(BlockDim);
    dim3 const gridDim((M + blockDim.x - 1) / blockDim.x);

    detail::sequential_with_shared_memory_kernel<<<gridDim, blockDim>>>(
        data_out, data_in);
  }
};

} // namespace batched_reduction_operator
