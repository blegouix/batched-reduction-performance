// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <iostream>

#include <mdspan/mdspan.hpp>

#include <benchmark/benchmark.h>

#include <batched_reduction_performance/batched_reduction_performance.hpp>

static constexpr std::size_t N = 1024;
static constexpr std::size_t M = 32;

void dummy_benchmark(benchmark::State &state) {
  std::array<double, N * M> mat_alloc;

  Kokkos::mdspan<double, Kokkos::extents<std::size_t, N, M>> data(
      mat_alloc.data(), N, M);
  for (std::size_t i = 0; i < data.extent(0); ++i) {
    for (std::size_t j = 0; j < data.extent(1); ++j) {
      data(i, j) = i * N + j;
      // std::cout << data(i, j) << " ";
    }
    // std::cout << "\n";
  }

  for (auto _ : state) {
    batched_reduction_kernel::dummy_kernel();
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0) * sizeof(double)));
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
