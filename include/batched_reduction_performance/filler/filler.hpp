// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cuda/std/mdspan>

#pragma once

namespace filler {

template <std::size_t M, std::size_t N, class Layout>
__global__ void fill_kernel(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>, Layout>
        data) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

  data(i, j) = static_cast<double>(i * M + j);
}

template <std::size_t BlockDim1, std::size_t BlockDim2, std::size_t M,
          std::size_t N, class Layout>
void fill(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>, Layout>
        data) {
  dim3 const blockDim(BlockDim1, BlockDim2);
  dim3 const gridDim((M + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);

  fill_kernel<<<gridDim, blockDim>>>(data);
}

} // namespace filler
