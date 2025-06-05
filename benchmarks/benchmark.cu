// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cuda/std/mdspan>

#include <benchmark/benchmark.h>

#include <batched_reduction_performance/batched_reduction_performance.hpp>

static constexpr std::size_t M = 131072;
static constexpr std::size_t N = 4096;

static constexpr std::size_t BlockDim1D = 256;
static constexpr std::size_t BlockDim2D_1 = 16;
static constexpr std::size_t BlockDim2D_2 = 16;

static_assert(M >= BlockDim1D &&
              "M has to be equal or greater than BlockDim1D");
static_assert(M >= BlockDim2D_1 * BlockDim2D_2 &&
              "M has to be equal or greater than BlockDim2D_1*BlockDim2D_2");

template <class BatchedReductionOperator, class Layout> class BatchedReductionBenchmark {
public:
  static void run(benchmark::State &state) {
    double *data_in_ptr = nullptr;
    cudaMalloc(&data_in_ptr, M * N * sizeof(double));

    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>, Layout> data_in(
        data_in_ptr);
    filler::fill<BlockDim2D_1, BlockDim2D_2>(data_in);
    // printer::print<BlockDim2D_1, BlockDim2D_2>(data_in);

    double *data_out_ptr = nullptr;
    cudaMalloc(&data_out_ptr, M * sizeof(double));

    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M>> data_out(
        data_out_ptr);

    for (auto _ : state) {
      BatchedReductionOperator::run(data_out, data_in);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) *
                            int64_t(M * N * sizeof(double)));

    // printer::print<BlockDim1D>(data_out);
    checker::check<BlockDim1D>(data_out, data_in);

    cudaFree(data_in_ptr);
    cudaFree(data_out_ptr);
  }
};

BENCHMARK(BatchedReductionBenchmark<
          batched_reduction_operator::Sequential<BlockDim1D>, cuda::std::layout_right>::run);
BENCHMARK(BatchedReductionBenchmark<
          batched_reduction_operator::Sequential<BlockDim1D>, cuda::std::layout_left>::run);
BENCHMARK(BatchedReductionBenchmark<
          batched_reduction_operator::CooperativeGroups, cuda::std::layout_right>::run);
BENCHMARK(BatchedReductionBenchmark<
          batched_reduction_operator::CooperativeGroups, cuda::std::layout_left>::run);

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
