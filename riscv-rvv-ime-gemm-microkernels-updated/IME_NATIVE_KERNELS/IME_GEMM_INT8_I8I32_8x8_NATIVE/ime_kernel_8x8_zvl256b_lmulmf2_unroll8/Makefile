CC=gcc
CFLAGS=-O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=c11 -Wall -Wno-unknown-pragmas

# Public SpacemiT documentation reserves non-1 IME LMUL values.
# Use ENABLE_MF2=1 to include LMUL mf2 native IME kernels.
ENABLE_MF2 ?= 0
ifeq ($(ENABLE_MF2),1)
CFLAGS += -DSPACEMIT_IME_ENABLE_MF2_NATIVE=1
endif
TARGET=bench
KERNEL=ime_kernel_8x8_zvl256b_lmulmf2_unroll8.c
BENCH=ime_bench.c
RVV_FALLBACK=rvv_fallback.c
all: $(TARGET)

$(TARGET): $(KERNEL) $(RVV_FALLBACK) $(BENCH)
	$(CC) $(CFLAGS) $(KERNEL) $(RVV_FALLBACK) $(BENCH) -o $(TARGET)

clean:
	rm -f $(TARGET)