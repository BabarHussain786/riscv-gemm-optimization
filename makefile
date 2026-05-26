# COMPILATION SUITE V2
#
# DON'T CHANGE THIS FILE
# this file takes only order in compilation  
#
# this file uses builder.sh script, resolve association host-target
# change setting in builder.sh file for compiler settings

RISCV_OPT = -march=rv64gcv -mabi=lp64d

TARGETS = smatmul_baseline \
          smatmul_loopinterchange_O0 \
          smatmul_loopinterchange_O1 \
          smatmul_loopinterchange_O2 \
          smatmul_loopinterchange_O3 \
          smatmul_pfor \
          smatmul_tiling \
          smatmul_tiling_v2 \
          smatmul_tiling_v3 \
          smatmul_tiling_v4 \
          smatmul_recursive \
          rvv_smatmul_recursive \
          rvv2_smatmul_recursive \
          rvv_smatmul_recursive_O3 \
          rvv2_smatmul_recursive_O3 \

all: $(TARGETS)

# compile with -O0
smatmul_baseline:
	./builder.sh x86_64 -O0 -o build/x86_64/smatmul_baseline smatmul_baseline.c
	./builder.sh riscv64_emu -O0 -o build/qemu/smatmul_baseline smatmul_baseline.c $(RISCV_OPT)
	./builder.sh riscv64 -O0 -o build/riscv64/smatmul_baseline smatmul_baseline.c $(RISCV_OPT)

smatmul_loopinterchange_O0:
	./builder.sh x86_64 -O0 -o build/x86_64/smatmul_loopinterchange_O0 smatmul_loopinterchange.c
	./builder.sh riscv64_emu -O0 -o build/qemu/smatmul_loopinterchange_O0 smatmul_loopinterchange.c $(RISCV_OPT)
	./builder.sh riscv64 -O0 -o build/riscv64/smatmul_loopinterchange_O0 smatmul_loopinterchange.c $(RISCV_OPT)

smatmul_loopinterchange_O1:
	./builder.sh x86_64 -O1 -o build/x86_64/smatmul_loopinterchange_O1 smatmul_loopinterchange.c
	./builder.sh riscv64_emu -O1 -o build/qemu/smatmul_loopinterchange_O1 smatmul_loopinterchange.c $(RISCV_OPT)
	./builder.sh riscv64 -O1 -o build/riscv64/smatmul_loopinterchange_O1 smatmul_loopinterchange.c $(RISCV_OPT)

smatmul_loopinterchange_O2:
	./builder.sh x86_64 -O2 -o build/x86_64/smatmul_loopinterchange_O2 smatmul_loopinterchange.c
	./builder.sh riscv64_emu -O2 -o build/qemu/smatmul_loopinterchange_O2 smatmul_loopinterchange.c $(RISCV_OPT)
	./builder.sh riscv64 -O2 -o build/riscv64/smatmul_loopinterchange_O2 smatmul_loopinterchange.c $(RISCV_OPT)

smatmul_loopinterchange_O3:
	./builder.sh x86_64 -O3 -o build/x86_64/smatmul_loopinterchange_O3 smatmul_loopinterchange.c
	./builder.sh riscv64_emu -O3 -o build/qemu/smatmul_loopinterchange_O3 smatmul_loopinterchange.c $(RISCV_OPT)
	./builder.sh riscv64 -O3 -o build/riscv64/smatmul_loopinterchange_O3 smatmul_loopinterchange.c $(RISCV_OPT)


rvv_smatmul_recursive_O3:
	./builder.sh x86_64 -O3 -o build/x86_64/rvv_smatmul_recursive_O3 rvv_smatmul_recursive.c -fopenmp
	./builder.sh riscv64_emu -O3 -o build/qemu/rvv_smatmul_recursive_O3 rvv_smatmul_recursive.c -fopenmp $(RISCV_OPT)
	./builder.sh riscv64 -O3 -o build/riscv64/rvv_smatmul_recursive_O3 rvv_smatmul_recursive.c -fopenmp $(RISCV_OPT)

rvv2_smatmul_recursive_O3:
	./builder.sh x86_64 -O3 -o build/x86_64/rvv2_smatmul_recursive_O3 rvv2_smatmul_recursive.c -fopenmp
	./builder.sh riscv64_emu -O3 -o build/qemu/rvv2_smatmul_recursive_O3 rvv2_smatmul_recursive.c -fopenmp $(RISCV_OPT)
	./builder.sh riscv64 -O3 -o build/riscv64/rvv2_smatmul_recursive_O3 rvv2_smatmul_recursive.c -fopenmp $(RISCV_OPT)

#smatmul_pfor:
#smatmul_tiling:
#smatmul_tiling_v2:
#smatmul_tiling_v3:
#smatmul_tiling_v4:
#smatmul_recursive:
#rvv_smatmul_recursive:
#rvv2_smatmul_recursive:
# compile target with -O2 (because the best is -O2 in this case, vectorization will do apart, not with -O3)
%: %.c
	./builder.sh x86_64 -O2 -o build/x86_64/$@ $< -fopenmp
	./builder.sh riscv64_emu -O2 -o build/qemu/$@ $< -fopenmp $(RISCV_OPT)
	./builder.sh riscv64 -O2 -o build/riscv64/$@ $< -fopenmp $(RISCV_OPT)

clean:
	rm -f build/x86_64/*
	rm -f build/qemu/*
	rm -f build/riscv64/*
