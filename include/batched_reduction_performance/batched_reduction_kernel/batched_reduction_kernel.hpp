// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cuda/std/mdspan>

#pragma once

namespace batched_reduction_kernel {

namespace detail {

template <std::size_t _M, std::size_t _N>
static __global__ void sequential_kernel(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, _N>> data_out,
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, _M, _N>>
        data_in) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // if (i < _N && j < _N) {
  double tmp = 0;
  // TOOD shared buffer for data_in[i]

  for (std::size_t j = 0; j < _M; ++j) {
    tmp += data_in(i, j);
  }

  data_out(i) = tmp;
  printf("%f ", data_out(i));
  // }
}

} // namespace detail

class Sequential {
public:
  template <std::size_t _M, std::size_t _N>
  static void
  run(cuda::std::mdspan<double, cuda::std::extents<std::size_t, _N>> data_out,
      cuda::std::mdspan<double, cuda::std::extents<std::size_t, _M, _N>>
          data_in) {
    dim3 blockDim(256);
    dim3 gridDim((_N + blockDim.x - 1) / blockDim.x);

    detail::sequential_kernel<<<gridDim, blockDim>>>(data_out, data_in);
    cudaDeviceSynchronize();
  }
};

} // namespace batched_reduction_kernel
