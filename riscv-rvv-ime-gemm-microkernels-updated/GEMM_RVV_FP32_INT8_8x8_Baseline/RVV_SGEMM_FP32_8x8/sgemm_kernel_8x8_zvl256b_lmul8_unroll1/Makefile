CC = gcc
CFLAGS = -O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=c11 -Wall -Wno-unknown-pragmas
CPPFLAGS = -DCNAME=sgemm_kernel_8x8_zvl256b_lmul8_unroll1
LDLIBS = -lm
TARGET = bench
KERNEL = sgemm_kernel_8x8_zvl256b_lmul8_unroll1.c
BENCH = sgemm_bench.c

all: $(TARGET)

$(TARGET): $(KERNEL) $(BENCH)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(KERNEL) $(BENCH) -o $(TARGET) $(LDLIBS)

clean:
	rm -f $(TARGET)