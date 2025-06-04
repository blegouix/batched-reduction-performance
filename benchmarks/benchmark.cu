// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cuda/std/mdspan>

#include <benchmark/benchmark.h>

#include <batched_reduction_performance/batched_reduction_performance.hpp>

static constexpr std::size_t M = 1024;
static constexpr std::size_t N = 32;

void dummy_benchmark(benchmark::State &state) {
  double *data_in_ptr = nullptr;
  cudaMalloc(&data_in_ptr, M * N * sizeof(double));

  cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>> data_in(
      data_in_ptr);
  filler::fill(data_in);
  printer::print(data_in);

  double *data_out_ptr = nullptr;
  cudaMalloc(&data_out_ptr, M * sizeof(double));

  cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> data_out(
      data_out_ptr);

  for (auto _ : state) {
    batched_reduction_kernel::Sequential::run(data_out, data_in);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0) * sizeof(double)));

  cudaFree(data_in_ptr);
  cudaFree(data_out_ptr);
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
