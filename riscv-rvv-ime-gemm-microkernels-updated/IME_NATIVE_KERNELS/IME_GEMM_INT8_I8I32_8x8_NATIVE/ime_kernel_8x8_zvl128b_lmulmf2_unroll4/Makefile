CC=gcc
CFLAGS=-O3 -march=rv64gcv_zvl128b -mabi=lp64d -std=c11 -Wall -Wno-unknown-pragmas

# Public SpacemiT documentation reserves non-1 IME LMUL values.
# Use EXPERIMENTAL_MF2=1 only for explicit research-board capability tests.
EXPERIMENTAL_MF2 ?= 0
ifeq ($(EXPERIMENTAL_MF2),1)
CFLAGS += -DSPACEMIT_IME_ENABLE_EXPERIMENTAL_MF2_NATIVE=1
endif
TARGET=bench
KERNEL=ime_kernel_8x8_zvl128b_lmulmf2_unroll4.c
RVV_FALLBACK=rvv_fallback.c
BENCH=ime_bench.c

all: $(TARGET)

$(TARGET): $(KERNEL) $(RVV_FALLBACK) $(BENCH)
	$(CC) $(CFLAGS) $(KERNEL) $(RVV_FALLBACK) $(BENCH) -o $(TARGET)

clean:
	rm -f $(TARGET)
