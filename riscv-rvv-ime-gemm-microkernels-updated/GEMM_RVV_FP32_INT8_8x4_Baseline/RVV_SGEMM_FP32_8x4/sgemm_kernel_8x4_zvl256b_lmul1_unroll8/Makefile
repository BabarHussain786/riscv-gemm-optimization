CC=gcc
CFLAGS=-O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=c11 -Wall -Wno-unknown-pragmas -DCNAME=sgemm_kernel_8x4_zvl256b_lmul1_unroll8
TARGET=bench
KERNEL=sgemm_kernel_8x4_zvl256b_lmul1_unroll8.c
BENCH=sgemm_bench.c

all: $(TARGET)

$(TARGET): $(KERNEL) $(BENCH)
	$(CC) $(CFLAGS) $(KERNEL) $(BENCH) -o $(TARGET)

clean:
	rm -f $(TARGET)
