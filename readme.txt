
# 🚀 Matrix Multiplication Optimization on RISC-V Architecture

This repository contains optimized implementations of matrix multiplication algorithms targeting RISC-V platforms of Banana Pi Board. The project demonstrates the impact of algorithmic and compiler-level optimizations on performance, including techniques like **loop interchange**, **recursive divide-and-conquer**, **tiling**, **OpenMP parallelization**, and **auto-vectorization** using Scalar and **RISC-V Vector Extension (RVV)**.

---

## 📌 Project Overview

Matrix multiplication is a core operation in numerous fields, from scientific computing to machine learning. Optimizing this computation on emerging open architectures like RISC-V is critical for building scalable and energy-efficient systems.

This project investigates and benchmarks various optimization techniques and evaluates their performance on real RISC-V hardware.

---

## 🛠️ Optimizations Implemented

| Optimization Technique       | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| 🔁 Loop Interchange         | Improves cache locality by reordering loops                                 |
| 🔄 Recursive Multiplication | Divides matrices into smaller blocks for better cache usage                 |
| 🧱 Tiling (Blocking)        | Splits matrices into tiles to enhance memory access patterns                |
| 💥 OpenMP Parallelization   | Enables multi-core parallel execution                                       |
| 🧠 RVV Autovectorization    | Leverages RISC-V Vector Extensions for data-level parallelism               |

---

## 📦 Repository Structure

```
📁 matrix-multiplication-riscv
├── src/                  # Source code in C
│   ├── matmul_baseline.c
│   ├── matmul_recursive.c
│   ├── matmul_tiling.c
│   └── ...
├── Makefile              # Build automation
├── README.md             # This file
└── report/               # Final report (PDF)
    └── Final Report.pdf
```

---

## 🧰 Setup Instructions

### 1. 🧑‍💻 Install RISC-V Toolchain

```bash
sudo apt-get install gcc-riscv64-linux-gnu
```

### 2. 📝 Compile Matrix Multiplication Code

```bash
riscv64-unknown-elf-gcc -O3 -march=rv64imac -mabi=lp64 -o matmul_rv src/matmul_recursive.c
```

> Replace `matmul_recursive.c` with your chosen implementation.

### 3. 🏃 Run the Executable on RISC-V

Run on actual hardware (e.g., Banana Pi) or via a RISC-V simulator:

```bash
./matmul_rv
```

### 4. 📈 Benchmark with `perf`

```bash
perf stat -e L1-dcache-loads,L1-dcache-load-misses ./matmul_rv
```

---

## 📊 Results Summary

| Method               | Matrix Size | Time (s) | Speedup |
|---------------------|-------------|----------|---------|
| Baseline            | 4096×4096   | 15755.16 | 1.0×    |
| Loop Interchange    | 4096×4096   | 379.62   | 41.5×   |
| Recursive (128)     | 4096×4096   | 1712.26  | 9.2×    |
| Tiled OpenMP        | 1024×1024   | 17.21    | 13.1×   |
| RVV2 Recursive       | 4096×4096   | ~        | 1.0–2.6×|

> Full benchmarking details and graphs are available in the [Final Report](report/Final%20Report.pdf).

---

## 🔍 Key Learnings

- **Compiler optimization flags matter**: `-O3` significantly boosts performance.
- **Algorithm-architecture synergy**: Recursive and tiling methods effectively utilize RISC-V memory hierarchies.
- **Parallelism is powerful**: OpenMP and RVV extensions scale well on multi-core RISC-V chips.

---

## 📚 References

- [RISC-V GCC Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [RISC-V Vector Extension (RVV)](https://github.com/riscv/riscv-v-spec)

---

## ✍️ Authors

- **Babar Hussain** – University of Salerno
- **Sergio Guastaferro** – University of Salerno

---

## 🙏 Acknowledgments

Special thanks to the **ISIS Lab** and the **University of Salerno – HPC Course** for providing hardware access and academic support.


