RISC-V Matrix Multiplication Optimization

This repository contains the implementation and evaluation of various matrix multiplication optimization techniques on RISC-V architecture.

## Overview
Matrix multiplication is a fundamental operation in many computational tasks. Optimizing its performance on emerging architectures like RISC-V is critical for advancing high-performance computing capabilities. This project presents a comprehensive evaluation of various matrix multiplication optimization techniques on RISC-V platforms, including loop interchange, recursive divide-and-conquer, OpenMP-based parallelization, and cache-aware tiling.

## Report
For detailed results and analysis, please refer to the [Final Report](docs/Final Report.pdf).

## Implementation
The matrix multiplication algorithm was implemented in C, with optimizations for loop interchange, recursion, and tiling. The code is available in the [src](src/) directory.

## Benchmarking
Benchmarking was conducted on the Banana Pi RISC-V board with a 64-bit RISC-V processor. Performance metrics include execution time and cache efficiency.

## Optimization Techniques
- **Loop Interchange**: Reorders nested loops to improve cache locality.
- **Recursive Algorithms**: Divides large matrices into smaller sub-matrices to simplify computation.
- **Tiling**: Breaks matrices into smaller blocks to improve memory access patterns and cache efficiency.
- **OpenMP Parallelization**: Utilizes multiple threads to parallelize the computation.

## Results
The results demonstrate substantial improvements in execution time and cache efficiency, with loop interchange and tiling achieving speedups up to 57.5× over the baseline naive implementation. Recursive algorithms also deliver significant gains when applied with suitable thresholds.


See in this order
(scalar)
1)smatmul_baseline.c
2)smatmul_loopinterchange.c
3)smatmul_pfor.c
4)smatmul_tiling.c  ( + version v2, v3, v4 )
5)smatmul_recursive.c

(vectorial)
6)rvv_smatmul_recursive.c  (LMUL=1)
7)rvv2_smatmul_recursive.c (LMUL=2)


------------------

If there was benchmark in execution, 
try to check in the screen name "benchmark"

1. Make a screen:
screen -S benchmark

2. detach
ctrl-a then d

3. Resume the screen:
screen -r benchmark


## GitHub Page
For more details and interactive content, visit the [GitHub Page](https://BabarHussain786.github.io/riscv-gemm-optimization).

## Acknowledgments
This project was supported by the University of Salerno and the Department of Computer Science. Special thanks to the High-performance computing course and ISIS Lab for providing access to the Banana Pi and Spacemit X60 RISC-V platforms.

## References
- J. Bennett, K. McKinley, Matrix multiplication and performance optimization: A comprehensive study, ACM Computing Surveys 41 (2008) 1–25.
- Z. Liu, X. Li, Optimizing matrix multiplication algorithms for large-scale systems, Journal of Parallel and Distributed Computing 74 (2014) 1187–1196.
- M. Harris, Optimizing parallel computing with simd and vector processing, International Journal of High Performance Computing Applications 24 (2010) 78–89.
- G. Singh, R. Kumar, Optimizing matrix operations with recursive algorithms on modern hardware architectures, Journal of Computer Science and Technology 30 (2015) 400–412.
- J. Kim, S. Kim, Tiling techniques for matrix multiplication optimization on multi-core systems, IEEE Transactions on Parallel and Distributed Systems 28 (2017) 2486–2495.
