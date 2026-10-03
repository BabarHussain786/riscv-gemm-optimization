#ifndef FAIR_GEMM_BENCH_H
#define FAIR_GEMM_BENCH_H
#include <stdint.h>
#include <stddef.h>
typedef struct {
    long m, n, k, full_n, n0;
    const int8_t *a, *b;
    int32_t *c;
    int8_t *ap, *bp;
    void *private_data;
} BenchTile;
typedef struct { double packing, kernel, output, boundary; } BenchPhases;
typedef struct {
    const char *name;
    int output_fused, boundary_fused;
    int (*supported)(void);
    int (*allocate)(BenchTile *);
    int (*pack)(BenchTile *);
    int (*execute)(BenchTile *, BenchPhases *, int);
    void (*release)(BenchTile *);
} BenchBackend;
extern const BenchBackend rvv_backend, ime_backend;
double bench_now(void);
void *bench_alloc(size_t count, size_t size);
#endif
