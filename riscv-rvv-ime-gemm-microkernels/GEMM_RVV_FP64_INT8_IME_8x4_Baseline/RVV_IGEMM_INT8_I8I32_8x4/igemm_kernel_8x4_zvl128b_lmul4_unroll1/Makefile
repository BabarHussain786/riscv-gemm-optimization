CC=gcc
CFLAGS=-O3 -march=rv64gcv_zvl128b -mabi=lp64d -std=c11 -Wall -Wno-unknown-pragmas
TARGET=bench
KERNEL=igemm_kernel_8x4_zvl128b_lmul4_unroll1_i8i32.c
BENCH=igemm_bench.c

all: $(TARGET)

$(TARGET): $(KERNEL) $(BENCH)
	$(CC) $(CFLAGS) $(KERNEL) $(BENCH) -o $(TARGET)

clean:
	rm -f $(TARGET)
