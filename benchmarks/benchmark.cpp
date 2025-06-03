// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <batched_reduction_performance/batched_reduction_performance.hpp>

#include <benchmark/benchmark.h>

void dummy_benchmark(benchmark::State &state) {
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
