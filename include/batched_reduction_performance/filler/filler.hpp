// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <iostream>

#include <cuda/std/mdspan>

#pragma once

namespace filler {

template <std::size_t _M, std::size_t _N>
__global__ void fill_kernel(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, _M, _N>> data) {
  std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < _M && j < _N) {
    data(i, j) = static_cast<double>(i * _N + j);
    // printf("%f ", static_cast<double>(data(i, j)));
  }
}

template <std::size_t _M, std::size_t _N>
void fill(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, _M, _N>> data) {
  dim3 blockDim(16, 16);
  dim3 gridDim((_M + blockDim.x - 1) / blockDim.x,
               (_N + blockDim.y - 1) / blockDim.y);

  fill_kernel<<<gridDim, blockDim>>>(data);
  cudaDeviceSynchronize();
}

} // namespace filler
