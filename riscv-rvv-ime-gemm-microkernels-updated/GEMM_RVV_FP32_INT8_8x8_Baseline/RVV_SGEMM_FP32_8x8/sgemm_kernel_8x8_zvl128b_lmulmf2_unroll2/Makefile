CC=gcc

CFLAGS=-O3 \
	-march=rv64gcv_zvl128b \
	-mabi=lp64d \
	-std=c11 \
	-Wall

TARGET=bench

KERNEL=sgemm_kernel_8x8_zvl128b_lmulmf2_unroll2.c
BENCH=sgemm_bench.c

all: $(TARGET)

$(TARGET): $(KERNEL) $(BENCH)
	$(CC) $(CFLAGS) $(KERNEL) $(BENCH) -o $(TARGET)

clean:
	rm -f $(TARGET)

