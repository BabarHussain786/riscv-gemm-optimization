/* Additive adapter: reuse the original dispatch packer without editing it. */
#include "bench.h"
#include <stdlib.h>
#include <riscv_vector.h>
typedef long BLASLONG;
typedef int8_t INPUT_T;
typedef int32_t OUTPUT_T;
#define OMP_KIND_INT8_RVV 1
#define OMP_KERNEL_MR 8
#define OMP_KERNEL_NR BENCH_RVV_NR
/* The existing helper packs then calls KERNEL_SYMBOL. This callback suppresses
 * only that last call, allowing the exact packer to be timed separately. */
static int pack_only(long m,long n,long k,int32_t alpha,int8_t *a,int8_t *b,
                     int32_t *c,long ldc) {
    (void)m;(void)n;(void)k;(void)alpha;(void)a;(void)b;(void)c;(void)ldc;
    return 0;
}
#define KERNEL_SYMBOL pack_only
#include BENCH_DISPATCH_SOURCE
#undef KERNEL_SYMBOL
extern int bench_rvv_entry(long,long,long,int32_t,int8_t*,int8_t*,int32_t*,long);
static int supported(void) { return __riscv_vsetvlmax_e8m1() == 32; }
static int allocate(BenchTile *t) {
    t->ap=bench_alloc((size_t)t->m*t->k,1);
    t->bp=bench_alloc((size_t)t->n*t->k,1);
    return !t->ap || !t->bp;
}
static int pack(BenchTile *t) {
    return call_packed_rvv_tile_kernel(t->m,t->n,t->k,t->n0,t->full_n,
        (int8_t*)t->a,(int8_t*)t->b,t->c,t->ap,t->bp);
}
static int execute(BenchTile *t,BenchPhases *p,int profile) {
    double start=profile?bench_now():0;
    int rc=bench_rvv_entry(t->m,t->n,t->k,1,t->ap,t->bp,
                         t->c+t->n0*t->m,t->m);
    if(profile) p->kernel+=bench_now()-start;
    return rc;
}
static void release(BenchTile *t) { free(t->ap);free(t->bp); }
const BenchBackend rvv_backend={"rvv",1,1,supported,allocate,pack,execute,release};
