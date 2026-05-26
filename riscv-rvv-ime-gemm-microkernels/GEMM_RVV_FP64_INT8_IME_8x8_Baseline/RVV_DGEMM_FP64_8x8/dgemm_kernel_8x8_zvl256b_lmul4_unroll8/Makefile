CC=gcc
# Still compiling with zvl256b target for your Banana Pi
CFLAGS=-O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=c11 -Wall -Wno-unknown-pragmas
TARGET=bench
# Updated to look for the zvl256b filename
KERNEL=dgemm_kernel_8x8_zvl256b_lmul4_unroll8.c
BENCH=dgemm_bench.c

all: $(TARGET)

$(TARGET): $(KERNEL) $(BENCH)
	$(CC) $(CFLAGS) $(KERNEL) $(BENCH) -o $(TARGET)

clean:
	rm -f $(TARGET)
