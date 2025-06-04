// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <iostream>

#include <cuda/std/mdspan>

#include <benchmark/benchmark.h>

#include <batched_reduction_performance/batched_reduction_performance.hpp>

// TODO restore .cpp extension

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
void filler(
    cuda::std::mdspan<double, cuda::std::extents<std::size_t, _M, _N>> data) {
  dim3 blockDim(16, 16);
  dim3 gridDim((_M + blockDim.x - 1) / blockDim.x,
               (_N + blockDim.y - 1) / blockDim.y);

  fill_kernel<<<gridDim, blockDim>>>(data);
  cudaDeviceSynchronize();
}

static constexpr std::size_t M = 32;
static constexpr std::size_t N = 1024;

void dummy_benchmark(benchmark::State &state) {
  double *mat_ptr = nullptr;
  cudaMalloc(&mat_ptr, M * N * sizeof(double));

  cuda::std::mdspan<double, cuda::std::extents<std::size_t, N, M>> mat(mat_ptr);
  filler(mat);

  for (auto _ : state) {
    batched_reduction_kernel::dummy_kernel();
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0) * sizeof(double)));

  cudaFree(mat_ptr);
}

std::size_t constexpr dummy_param = 128;

BENCHMARK(dummy_benchmark)->Arg(dummy_param);

int main(int argc, char **argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  {
    ::benchmark::RunSpecifiedBenchmarks();
  }
  ::benchmark::Shutdown();
  return 0;
}
