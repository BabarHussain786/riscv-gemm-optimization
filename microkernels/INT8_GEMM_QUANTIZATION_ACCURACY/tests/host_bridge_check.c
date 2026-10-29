/*
 * Host-only bridge contract checks. The two stub backends below never execute
 * RVV/IME instructions and do not establish native kernel correctness.
 *
 * On a Linux host, run from INT8_GEMM_QUANTIZATION_ACCURACY:
 *   cc -std=c11 -Wall -Wextra -Werror -fPIC -I ../benchmarking/src \
 *      kernel_bridge.c tests/host_bridge_check.c -o /tmp/qa-host-bridge-check
 *   /tmp/qa-host-bridge-check
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "bench.h"
#include <assert.h>
#include <limits.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int qa_abi_version(void);
int qa_run(int, int, long, long, long, const int8_t *, const int8_t *, int32_t *);

static int allow_hardware = 1;
static int fail_stage = 0;
static int release_count = 0;

static int stub_supported(void)
{
    return allow_hardware;
}

static int stub_allocate(BenchTile *tile)
{
    tile->ap = malloc(1);
    tile->bp = malloc(1);
    assert(tile->ap != NULL && tile->bp != NULL);
    return fail_stage == 1;
}

static int stub_pack(BenchTile *tile)
{
    assert(tile->m == 8 && tile->n == 4 && tile->k == 8);
    assert(tile->full_n == 4 && tile->n0 == 0);
    return fail_stage == 2;
}

static int stub_execute(BenchTile *tile, BenchPhases *phases, int profile)
{
    assert(phases != NULL && profile == 0);
    for (size_t i = 0; i < 32; ++i) assert(tile->c[i] == 0);
    tile->c[0] = 42;
    return fail_stage == 3;
}

static void stub_release(BenchTile *tile)
{
    free(tile->ap);
    free(tile->bp);
    ++release_count;
}

const BenchBackend rvv_backend = {
    "stub-rvv", 1, 1, stub_supported, stub_allocate, stub_pack,
    stub_execute, stub_release
};
const BenchBackend ime_backend = {
    "stub-ime", 0, 0, stub_supported, stub_allocate, stub_pack,
    stub_execute, stub_release
};

static void check_affinity(const cpu_set_t *expected)
{
    cpu_set_t actual;
    assert(sched_getaffinity(0, sizeof(actual), &actual) == 0);
    assert(CPU_EQUAL(expected, &actual));
}

static void check_allocations(void)
{
    const size_t counts[] = {13, 0, 4};
    const size_t sizes[] = {3, 4, 0};
    assert(bench_alloc(SIZE_MAX, 2) == NULL);
    for (size_t example = 0; example < 3; ++example) {
        size_t bytes = counts[example] * sizes[example];
        unsigned char *allocation = bench_alloc(counts[example], sizes[example]);
        if (bytes == 0) bytes = 64;
        assert(allocation != NULL && (uintptr_t)allocation % 64 == 0);
        for (size_t i = 0; i < bytes; ++i) assert(allocation[i] == 0);
        free(allocation);
    }
}

int main(void)
{
    cpu_set_t saved;
    int cpu = -1;
    int unavailable = -1;
    int8_t a[64] = {0};
    int8_t b[32] = {0};
    int32_t c[32];

    assert(sched_getaffinity(0, sizeof(saved), &saved) == 0);
    for (int candidate = 0; candidate < CPU_SETSIZE; ++candidate) {
        if (CPU_ISSET(candidate, &saved) && cpu < 0) cpu = candidate;
        if (!CPU_ISSET(candidate, &saved) && unavailable < 0)
            unavailable = candidate;
    }
    assert(cpu >= 0);
    assert(qa_abi_version() == 1);
    check_allocations();

    assert(qa_run(2, cpu, 8, 4, 8, a, b, c) == 1);
    assert(qa_run(0, cpu, 7, 4, 8, a, b, c) == 2);
    assert(qa_run(0, cpu, 8, 4, 131072, a, b, c) == 2);
    assert(qa_run(0, cpu, 8, 4, 8, NULL, b, c) == 3);
    assert(qa_run(0, cpu, LONG_MAX - 7, 4, 8, a, b, c) == 3);
    assert(qa_run(0, -1, 8, 4, 8, a, b, c) == 4);
    assert(qa_run(0, CPU_SETSIZE, 8, 4, 8, a, b, c) == 4);
    if (unavailable >= 0) {
        assert(qa_run(0, unavailable, 8, 4, 8, a, b, c) == 4);
        check_affinity(&saved);
    }

    memset(c, 0x7f, sizeof(c));
    allow_hardware = 0;
    assert(qa_run(0, cpu, 8, 4, 8, a, b, c) == 7);
    assert(release_count == 0);
    for (size_t i = 0; i < 32; ++i) assert(c[i] == 0);
    check_affinity(&saved);

    allow_hardware = 1;
    for (int stage = 1; stage <= 3; ++stage) {
        fail_stage = stage;
        assert(qa_run(0, cpu, 8, 4, 8, a, b, c) == stage + 7);
        assert(release_count == stage);
        check_affinity(&saved);
    }
    fail_stage = 0;
    for (int backend = 0; backend < 2; ++backend) {
        memset(c, 0x7f, sizeof(c));
        assert(qa_run(backend, cpu, 8, 4, 8, a, b, c) == 0);
        assert(c[0] == 42);
        for (size_t i = 1; i < 32; ++i) assert(c[i] == 0);
        check_affinity(&saved);
    }
    puts("Bridge host stub validation: OK (no native kernels executed)");
    return 0;
}
