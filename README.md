# batched-reduction-performance
Compare performance for several batched reduction algorithms and different memory organizations using CUDA

## Usage
```
git clone --recurse-submodules git@github.com:blegouix/batched-reduction-performance.git && cd batched-reduction-performance/
cmake -B build/ -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_BUILD_TYPE=Release -DALLOW_UNCOMPLETE_WARP=OFF && cd build/
make -j8 && ./benchmarks/batched-reduction-performance
```
