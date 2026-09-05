CC ?= gcc
CFLAGS ?= -O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=c11 -Wall -Wno-unknown-pragmas
LDLIBS ?= -lm
TARGET := bench
KERNEL := dgemm_kernel_8x4_zvl256b_lmul2_unroll8.c
BENCH := dgemm_bench.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(KERNEL) $(BENCH)
	$(CC) $(CFLAGS) $(KERNEL) $(BENCH) $(LDLIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)