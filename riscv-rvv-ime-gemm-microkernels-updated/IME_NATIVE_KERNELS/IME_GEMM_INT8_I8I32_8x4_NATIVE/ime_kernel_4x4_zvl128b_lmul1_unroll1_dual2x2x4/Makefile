CC=gcc

CFLAGS=-O3 -march=rv64gcv_zvl128b -mabi=lp64d -std=c11 -Wall -Wno-unknown-pragmas

TARGET=bench
KERNEL=ime_kernel_4x4_zvl128b_lmul1_unroll1_dual2x2x4.c
RVV_FALLBACK=rvv_fallback.c
BENCH=ime_bench.c

all: $(TARGET)

$(TARGET): $(KERNEL) $(RVV_FALLBACK) $(BENCH)
	$(CC) $(CFLAGS) $(KERNEL) $(RVV_FALLBACK) $(BENCH) -o $(TARGET)

clean:
	rm -f $(TARGET)
