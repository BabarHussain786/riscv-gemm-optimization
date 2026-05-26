#!/bin/bash

echo "Benchsuite V1"

# baseline O0, 2x2 -> 4096x4096
echo "baseline O0, 2x2 -> 4096x4096"
./testsuite.sh build/riscv64/smatmul_baseline 2 4096 >> report/baseline_2_4096.txt

# loopinterchange O0, 2x2 -> 4096x4096
echo "loopinterchange O0, 2x2 -> 4096x4096"
./testsuite.sh build/riscv64/smatmul_loopinterchange_O0 2 4096 >> report/loopinterchage_O0.txt


# loop interchange O1, 2x2 -> 4096x4096
echo "loop interchange O1, 2x2 -> 4096x4096"
./testsuite.sh build/riscv64/smatmul_loopinterchange_O1 2 4096 >> report/loopinterchage_O1_010625.txt

# loop interchange O2, 2x2 -> 4096x4096
echo "loop interchange O2, 2x2 -> 4096x4096"
./testsuite.sh build/riscv64/smatmul_loopinterchange_O2 2 4096 >> report/loopinterchage_O2_010625.txt

# loop interchange O3, 2x2 -> 4096x4096
echo "loop interchange O3, 2x2 -> 4096x4096"
./testsuite.sh build/riscv64/smatmul_loopinterchange_O3 2 4096 >> report/loopinterchage_O3_010625.txt

# parallel loop O2, 2x2 -> 4096x4096
echo "parallel loop O2, 2x2 -> 4096x4096"
./testsuite.sh build/riscv64/smatmul_pfor 2 4096 >> report/pfor_010625.txt

# tiling O2, tile=64, 64x64 -> 4096x4096
echo "tiling O2, tile=64, 64x64 -> 4096x4096"
./testsuite.sh build/riscv64/smatmul_tiling 64 4096 >> report/tiling_010625.txt


# recursive O2, threshold=64  64x64 -> 4096x4096
echo "recursive O2, threshold=64  64x64 -> 4096x4096"
./testsuite.sh build/riscv64/smatmul_recursive 64 4096 64 >> report/recursive_t64_070625.txt

# recursive O2, threshold=128  128x128 -> 4096x4096
echo "recursive O2, threshold=128  128x128 -> 4096x4096"
./testsuite.sh build/riscv64/smatmul_recursive 128 4096 128 >> report/recursive_t128_070625.txt



### LMUL=1
echo "LMUL = 1"

# rvv_smatmul_recursive     threshold=64  64x64 -> 4096x4096
echo "rvv_smatmul_recursive     threshold=64  64x64 -> 4096x4096"
./testsuite.sh build/riscv64/rvv_smatmul_recursive 64 4096 64 >> report/rvv_t64_090625.txt

# rvv_smatmul_recursive     threshold=128  128x128 -> 4096x4096
echo "rvv_smatmul_recursive     threshold=128  128x128 -> 4096x4096"
./testsuite.sh build/riscv64/rvv_smatmul_recursive 128 4096 128 >> report/rvv_t128_090625.txt

# rvv_smatmul_recursive_O3  threshold=64  64x64 -> 4096x4096
echo "rvv_smatmul_recursive_O3  threshold=64  64x64 -> 4096x4096"
./testsuite.sh build/riscv64/rvv_smatmul_recursive_O3 64 4096 64 >> report/rvv_O3_t64_090625.txt

# rvv_smatmul_recursive_O3  threshold=128  128x128 -> 4096x4096
echo "rvv_smatmul_recursive_O3  threshold=128  128x128 -> 4096x4096"
./testsuite.sh build/riscv64/rvv_smatmul_recursive_O3 128 4096 128 >> report/rvv_O3_t128_090625.txt



### LMUL=2
echo "LMUL = 2"

# rvv2_smatmul_recursive     threshold=64  64x64 -> 4096x4096
echo "rvv2_smatmul_recursive     threshold=64  64x64 -> 4096x4096"
./testsuite.sh build/riscv64/rvv2_smatmul_recursive 64 4096 64 >> report/rvv2_t64_090625.txt

# rvv2_smatmul_recursive     threshold=128  128x128 -> 4096x4096
echo "rvv2_smatmul_recursive     threshold=128  128x128 -> 4096x4096"
./testsuite.sh build/riscv64/rvv2_smatmul_recursive 128 4096 128 >> report/rvv2_t128_090625.txt

# rvv2_smatmul_recursive_O3  threshold=64  64x64 -> 4096x4096
echo "rvv2_smatmul_recursive_O3  threshold=64  64x64 -> 4096x4096"
./testsuite.sh build/riscv64/rvv2_smatmul_recursive_O3 64 4096 64 >> report/rvv2_O3_t64_090625.txt

# rvv2_smatmul_recursive_O3  threshold=128  128x128 -> 4096x4096
echo "rvv2_smatmul_recursive_O3  threshold=128  128x128 -> 4096x4096"
./testsuite.sh build/riscv64/rvv2_smatmul_recursive_O3 128 4096 128 >> report/rvv2_O3_t128_090625.txt
