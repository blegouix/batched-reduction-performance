// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cuda/std/mdspan>

#pragma once

namespace filler {

template <std::size_t M, std::size_t N>
__global__ void fill_kernel(
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
    data(i, j) = static_cast<double>(i * M + j);
  }
}

template <std::size_t M, std::size_t N>
void fill(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>> data) {
  dim3 const blockDim(16, 16);
  dim3 const gridDim((M + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);

  fill_kernel<<<gridDim, blockDim>>>(data);
}

} // namespace filler
