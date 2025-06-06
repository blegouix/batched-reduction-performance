// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <iostream>

#include <cuda/std/mdspan>

#pragma once

namespace checker {

template <std::size_t M, std::size_t N, class Layout>
__global__ void check_kernel(
    bool *mismatch,
    cuda::std::mdspan<bool, cuda::std::extents<std::size_t, M>> buffer,
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> reduced_data,
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>, Layout>
        data) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  double sum = 0;
  for (std::size_t j = 0; j < N; ++j) {
    sum += data(i, j);
  }

  if (fabs(reduced_data(i) - sum) > 1e-6) {
    *mismatch = true;
  }
}

template <std::size_t BlockDim, std::size_t M, std::size_t N, class Layout>
void check(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> reduced_data,
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>, Layout>
        data) {
  dim3 const blockDim(BlockDim);
  dim3 const gridDim((M + blockDim.x - 1) / blockDim.x);

  bool h_mismatch = false;
  bool *mismatch;
  cudaMalloc(&mismatch, sizeof(bool));
  cudaMemcpy(mismatch, &h_mismatch, sizeof(bool), cudaMemcpyHostToDevice);

  double *buffer_ptr = nullptr;
  cudaMalloc(&buffer_ptr, M * sizeof(bool));
  cuda::std::mdspan<bool, cuda::std::extents<std::size_t, M>> buffer;

  check_kernel<<<gridDim, blockDim>>>(mismatch, buffer, reduced_data, data);

  cudaMemcpy(&h_mismatch, mismatch, sizeof(bool), cudaMemcpyDeviceToHost);

  if (h_mismatch) {
    std::cout << "Error: the result of the reduction is not correct!\n";
  }
}

} // namespace checker
