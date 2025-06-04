// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cuda/std/mdspan>

#pragma once

namespace batched_reduction_kernel {

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
    // TODO shared buffer for data_in[i]

    for (std::size_t j = 0; j < N; ++j) {
      tmp += data_in(i, j);
    }

    data_out(i) = tmp;
  }
}

} // namespace detail

class Sequential {
public:
  template <std::size_t M, std::size_t N>
  static void
  run(cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> data_out,
      cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>>
          data_in) {
    dim3 const blockDim(256);
    dim3 const gridDim((M + blockDim.x - 1) / blockDim.x);

    detail::sequential_kernel<<<gridDim, blockDim>>>(data_out, data_in);
  }
};

} // namespace batched_reduction_kernel
