/*
 * Small, synchronous ABI for the existing K1 INT8 BenchBackend adapters.
 *
 * Canonical input: A[k*m+i], B[k*n+j]. Output: C[j*m+i].
 * Backend 0 is the standalone RVV 8x4 mf8/unroll2 kernel; backend 1 is the
 * native IME 8x4 LMUL1/unroll1 kernel. C is reset before every valid call.
 * This file contains no GEMM implementation and never substitutes a reference
 * backend when hardware execution is unavailable.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "bench.h"
#include <limits.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    QA_OK = 0,
    QA_INVALID_BACKEND = 1,
    QA_INVALID_DIMENSIONS = 2,
    QA_INVALID_BUFFER = 3,
    QA_INVALID_CPU = 4,
    QA_AFFINITY_READ_FAILED = 5,
    QA_AFFINITY_PIN_FAILED = 6,
    QA_UNSUPPORTED_HARDWARE = 7,
    QA_ALLOCATION_FAILED = 8,
    QA_PACK_FAILED = 9,
    QA_KERNEL_FAILED = 10,
    QA_AFFINITY_RESTORE_FAILED = 11
};

/* The original adapters use these two services, without the benchmark main. */
void *bench_alloc(size_t count, size_t size)
{
    void *allocation = NULL;
    size_t bytes;
    if (size != 0 && count > SIZE_MAX / size) return NULL;
    bytes = count * size;
    if (bytes == 0) bytes = 64;
    if (posix_memalign(&allocation, 64, bytes) != 0) return NULL;
    memset(allocation, 0, bytes);
    return allocation;
}

double bench_now(void)
{
    struct timespec stamp;
    if (clock_gettime(CLOCK_MONOTONIC, &stamp) != 0) return 0.0;
    return (double)stamp.tv_sec + (double)stamp.tv_nsec * 1e-9;
}

int qa_abi_version(void)
{
    return 1;
}

/* Also bound pointer differences and the signed long indexing in the adapters. */
static int matrix_bytes(long rows, long columns, size_t item_size, size_t *bytes)
{
    size_t elements;
    if (rows <= 0 || columns <= 0 || item_size == 0) return 0;
    if ((size_t)rows > SIZE_MAX / (size_t)columns) return 0;
    elements = (size_t)rows * (size_t)columns;
    if (elements > (size_t)LONG_MAX || elements > SIZE_MAX / item_size ||
        elements > (size_t)PTRDIFF_MAX / item_size) return 0;
    *bytes = elements * item_size;
    return 1;
}

/*
 * Return zero only after a successful kernel call and affinity restoration.
 * Nonzero values are stable failure categories from the enum above.
 * Callers must pass allocated, nonoverlapping buffers of the documented sizes.
 */
int qa_run(int backend, int cpu, long m, long n, long k,
           const int8_t *a, const int8_t *b, int32_t *c)
{
    const BenchBackend *engine;
    BenchTile tile = {0};
    BenchPhases phases = {0};
    cpu_set_t saved_affinity;
    cpu_set_t requested_affinity;
    size_t a_bytes, b_bytes, c_bytes;
    int workspace_started = 0;
    int result = QA_OK;

    if (backend != 0 && backend != 1) return QA_INVALID_BACKEND;
    /* These conditions exclude every scalar boundary in the IME adapter.
     * Even full-range signed INT8 has |dot| <= k*16384 <= INT32_MAX. */
    if (m <= 0 || n <= 0 || k <= 0 || m % 8 || n % 4 || k % 8 ||
        k > 131071) return QA_INVALID_DIMENSIONS;
    if (a == NULL || b == NULL || c == NULL ||
        !matrix_bytes(m, k, sizeof(*a), &a_bytes) ||
        !matrix_bytes(n, k, sizeof(*b), &b_bytes) ||
        !matrix_bytes(m, n, sizeof(*c), &c_bytes)) return QA_INVALID_BUFFER;
    (void)a_bytes;
    (void)b_bytes;
    if (cpu < 0 || cpu >= CPU_SETSIZE) return QA_INVALID_CPU;

    memset(c, 0, c_bytes);
    if (sched_getaffinity(0, sizeof(saved_affinity), &saved_affinity) != 0)
        return QA_AFFINITY_READ_FAILED;
    /* Keep execution within the caller's original permitted CPU selection. */
    if (!CPU_ISSET(cpu, &saved_affinity)) {
        result = QA_INVALID_CPU;
        goto restore_affinity;
    }
    CPU_ZERO(&requested_affinity);
    CPU_SET(cpu, &requested_affinity);
    if (sched_setaffinity(0, sizeof(requested_affinity), &requested_affinity) != 0 ||
        sched_getcpu() != cpu) {
        result = QA_AFFINITY_PIN_FAILED;
        goto restore_affinity;
    }

    engine = backend == 0 ? &rvv_backend : &ime_backend;
    if (!engine->supported()) {
        result = QA_UNSUPPORTED_HARDWARE;
        goto restore_affinity;
    }
    tile.m = m;
    tile.n = n;
    tile.k = k;
    tile.full_n = n;
    tile.n0 = 0;
    tile.a = a;
    tile.b = b;
    tile.c = c;
    /* release() also handles partially allocated, zero-initialized tiles. */
    workspace_started = 1;
    if (engine->allocate(&tile) != 0) {
        result = QA_ALLOCATION_FAILED;
        goto release_workspace;
    }
    if (engine->pack(&tile) != 0) {
        result = QA_PACK_FAILED;
        goto release_workspace;
    }
    if (engine->execute(&tile, &phases, 0) != 0) result = QA_KERNEL_FAILED;

release_workspace:
    if (workspace_started) engine->release(&tile);
restore_affinity:
    if (sched_setaffinity(0, sizeof(saved_affinity), &saved_affinity) != 0)
        result = QA_AFFINITY_RESTORE_FAILED;
    return result;
}
