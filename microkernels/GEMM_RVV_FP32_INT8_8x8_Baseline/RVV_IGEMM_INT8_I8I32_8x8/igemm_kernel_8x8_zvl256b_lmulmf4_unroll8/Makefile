CC = gcc
CFLAGS = -O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=c11 -Wall -Wno-unknown-pragmas
CPPFLAGS = -DCNAME=igemm_kernel_8x8_zvl256b_lmulmf4_unroll8_i8i32
LDLIBS = -lm
TARGET = bench
KERNEL = igemm_kernel_8x8_zvl256b_lmulmf4_unroll8_i8i32.c
BENCH = igemm_bench.c

all: $(TARGET)

$(TARGET): $(KERNEL) $(BENCH)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(KERNEL) $(BENCH) -o $(TARGET) $(LDLIBS)

clean:
	rm -f $(TARGET)