// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <iostream>

#include <cuda/std/mdspan>

#pragma once

namespace printer {

template <std::size_t M>
__global__ void print_1d_kernel(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> data) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

#if defined ALLOW_UNCOMPLETE_WARP
  if (i < M) {
#else
  {
    static_assert(M % 32 == 0, "Uncomplete warps are not allowed, fix the "
                               "problem sizes or enable ALLOW_UNCOMPLETE_WARP");
#endif
    printf("%f ", static_cast<double>(data(i)));
  }
}

template <std::size_t M, std::size_t N>
__global__ void print_2d_kernel(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>> data) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

#if defined ALLOW_UNCOMPLETE_WARP
  if (i < M && j < N) {
#else
  {
    static_assert(M % 32 == 0 && N % 32 == 0,
                  "Uncomplete warps are not allowed, fix the problem sizes or "
                  "enable ALLOW_UNCOMPLETE_WARP");
#endif
    printf("%f ", static_cast<double>(data(i, j)));
  }
}

template <std::size_t BlockDim, std::size_t M>
void print(cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> data) {
  dim3 const blockDim(BlockDim);
  dim3 const gridDim((M + blockDim.x - 1) / blockDim.x);

  print_1d_kernel<<<gridDim, blockDim>>>(data);
}

template <std::size_t BlockDim1, std::size_t BlockDim2, std::size_t M,
          std::size_t N>
void print(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>> data) {
  dim3 const blockDim(BlockDim1, BlockDim2);
  dim3 const gridDim((M + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);

  print_2d_kernel<<<gridDim, blockDim>>>(data);
}

} // namespace printer
