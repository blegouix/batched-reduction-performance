// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cuda/std/mdspan>

#pragma once

namespace batched_reduction_kernel {

namespace detail {

template <std::size_t M, std::size_t N>
static __global__ void sequential_kernel(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, N>> data_out,
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>> data_in) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // if (i < N && j < N) {
  double tmp = 0;
  // TOOD shared buffer for data_in[i]

  for (std::size_t j = 0; j < M; ++j) {
    tmp += data_in(i, j);
  }

  data_out(i) = tmp;
  printf("%f ", data_out(i));
  // }
}

} // namespace detail

class Sequential {
public:
  template <std::size_t M, std::size_t N>
  static void
  run(cuda::std::mdspan<double, cuda::std::extents<std::size_t, N>> data_out,
      cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>>
          data_in) {
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    detail::sequential_kernel<<<gridDim, blockDim>>>(data_out, data_in);
    cudaDeviceSynchronize();
  }
};

} // namespace batched_reduction_kernel
