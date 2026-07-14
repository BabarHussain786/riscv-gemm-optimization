CC = gcc

# Compiler flags targeting RV64 with Vector Extension (GCV) and 128-bit VLEN
CFLAGS = -O3 \
         -march=rv64gcv_zvl256b \
         -mabi=lp64d \
         -std=c11 \
         -Wall

TARGET = bench
KERNEL=sgemm_kernel_8x8_zvl256b_lmul8_unroll8.c
BENCH = sgemm_bench.c

all: $(TARGET)

$(TARGET): $(KERNEL) $(BENCH)
	$(CC) $(CFLAGS) $(KERNEL) $(BENCH) -o $(TARGET)

clean:
	rm -f $(TARGET)

