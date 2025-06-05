// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cub/block/block_reduce.cuh>
#include <cuda/std/mdspan>

#include <benchmark/benchmark.h>

#include <batched_reduction_performance/batched_reduction_performance.hpp>

static constexpr std::size_t BlockDim1D = 64;
static constexpr std::size_t BlockDim2D_1 = 4;
static constexpr std::size_t BlockDim2D_2 = 4;

template <std::size_t M, std::size_t N, class BatchedReductionOperator,
          class Layout>
class BatchedReductionBenchmark {
  static_assert(M >= BlockDim1D &&
                "M has to be equal or greater than BlockDim1D");
  static_assert(M >= BlockDim2D_1 * BlockDim2D_2 &&
                "M has to be equal or greater than BlockDim2D_1*BlockDim2D_2");
  static_assert(
      N <= 1024 &&
      "N has to be at most 1024 which is the number of threads per SM");

public:
  static void run(benchmark::State &state) {
    double *data_in_ptr = nullptr;
    cudaMalloc(&data_in_ptr, M * N * sizeof(double));

    cuda::std::mdspan<double, cuda::std::extents<std::size_t, M, N>, Layout>
        data_in = wrapper::wrap<M, N, Layout>(data_in_ptr);
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

#define BENCHMARKS(M, N)                                                         \
  BENCHMARK(BatchedReductionBenchmark<                                           \
            M, N, batched_reduction_operator::Sequential<BlockDim1D>,            \
            cuda::std::layout_right>::run);                                      \
  BENCHMARK(BatchedReductionBenchmark<                                           \
            M, N, batched_reduction_operator::Sequential<BlockDim1D>,            \
            cuda::std::layout_left>::run);                                       \
  BENCHMARK(BatchedReductionBenchmark<                                           \
            M, N, batched_reduction_operator::Sequential<BlockDim1D>,            \
            cuda::std::layout_stride>::run);                                     \
  /*                                                                             \
    BENCHMARK(                                                                   \
        BatchedReductionBenchmark<M, N,                                          \
                                  batched_reduction_operator::CooperativeGroups, \
                                  cuda::std::layout_right>::run);                \
    BENCHMARK(                                                                   \
        BatchedReductionBenchmark<M, N,                                          \
                                  batched_reduction_operator::CooperativeGroups, \
                                  cuda::std::layout_left>::run);                 \
    BENCHMARK(                                                                   \
        BatchedReductionBenchmark<M, N,                                          \
                                  batched_reduction_operator::CooperativeGroups, \
                                  cuda::std::layout_stride>::run);               \
  */                                                                             \
  BENCHMARK(                                                                     \
      BatchedReductionBenchmark<M, N,                                            \
                                batched_reduction_operator::CUBBlockReduction<   \
                                    cub::BLOCK_REDUCE_WARP_REDUCTIONS>,          \
                                cuda::std::layout_right>::run);                  \
  BENCHMARK(                                                                     \
      BatchedReductionBenchmark<M, N,                                            \
                                batched_reduction_operator::CUBBlockReduction<   \
                                    cub::BLOCK_REDUCE_WARP_REDUCTIONS>,          \
                                cuda::std::layout_left>::run);                   \
  BENCHMARK(                                                                     \
      BatchedReductionBenchmark<M, N,                                            \
                                batched_reduction_operator::CUBBlockReduction<   \
                                    cub::BLOCK_REDUCE_WARP_REDUCTIONS>,          \
                                cuda::std::layout_stride>::run);                 \
  BENCHMARK(BatchedReductionBenchmark<                                           \
            M, N,                                                                \
            batched_reduction_operator::CUBBlockReduction<                       \
                cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>,                      \
            cuda::std::layout_right>::run);                                      \
  BENCHMARK(BatchedReductionBenchmark<                                           \
            M, N,                                                                \
            batched_reduction_operator::CUBBlockReduction<                       \
                cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>,                      \
            cuda::std::layout_left>::run);                                       \
  BENCHMARK(BatchedReductionBenchmark<                                           \
            M, N,                                                                \
            batched_reduction_operator::CUBBlockReduction<                       \
                cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>,                      \
            cuda::std::layout_stride>::run);

/*
BENCHMARKS(65536, 64);
BENCHMARKS(65536, 128);
BENCHMARKS(65536, 256);
BENCHMARKS(65536, 512);
BENCHMARKS(65536, 1024);

BENCHMARKS(4096, 1024);
BENCHMARKS(8192, 1024);
BENCHMARKS(16384, 1024);
BENCHMARKS(32768, 1024);
BENCHMARKS(65536, 1024);
*/

BENCHMARKS(64, 1024);
BENCHMARKS(1024, 1024);
BENCHMARKS(65536, 1024);

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
