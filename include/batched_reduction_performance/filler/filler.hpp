// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cuda/std/mdspan>

#pragma once

namespace filler {

template <std::size_t M, std::size_t N>
__global__ void fill_kernel(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>> data) {
  std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < M && j < N) {
    data(i, j) = static_cast<double>(i * N + j);
  }
}

template <std::size_t M, std::size_t N>
void fill(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>> data) {
  dim3 blockDim(16, 16);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y);

  fill_kernel<<<gridDim, blockDim>>>(data);
  cudaDeviceSynchronize();
}

} // namespace filler
