CC=gcc
CFLAGS=-O3 -march=rv64gcv_zvl128b -mabi=lp64d -std=c11 -Wall -Wno-unknown-pragmas
TARGET=bench
# Pointing directly to the unroll2 C file
KERNEL=dgemm_kernel_8x8_zvl128b_lmul8_unroll2.c
BENCH=dgemm_bench.c

all: $(TARGET)

$(TARGET): $(KERNEL) $(BENCH)
	$(CC) $(CFLAGS) $(KERNEL) $(BENCH) -o $(TARGET)

clean:
	rm -f $(TARGET)
