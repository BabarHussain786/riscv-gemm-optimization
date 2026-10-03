/* Include the selected, UNMODIFIED IME translation unit to expose its static
 * packing/compute/scatter helpers. Compile its original fallback separately.
 * The public legacy driver is retained, but is not used by this benchmark. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "bench.h"
#include BENCH_IME_SOURCE
typedef struct { long km, mb, nb; size_t as, bs; int8_t *compact_b; } ImeWorkspace;
static int supported(void) {
    size_t vl;
    __asm__ __volatile__("li t0,-1\n\tvsetvli %0,t0,e8,m1,ta,ma" : "=r"(vl) :: "t0");
    return vl==32 && cpu_has_ime(sched_getcpu()) && detect_ime_profile()==IME_PROFILE_A60;
}
static int allocate(BenchTile *t) {
    ImeWorkspace *w=calloc(1,sizeof(*w));
    if(!w) return 1;
    t->private_data=w;
    w->km=t->k/(8*UNROLL_K)*(8*UNROLL_K);
    w->mb=t->m/IME_MR; w->nb=t->n/IME_NR;
    w->as=(size_t)IME_MR*w->km; w->bs=b_panel_bytes(IME_PROFILE_A60,w->km);
    t->ap=bench_alloc((size_t)w->mb*w->as,1);
    t->bp=bench_alloc((size_t)w->nb*w->bs,1);
    w->compact_b=bench_alloc((size_t)t->n*t->k,1);
    return !t->ap || !t->bp || !w->compact_b;
}
static int pack(BenchTile *t) {
    ImeWorkspace *w=t->private_data;
    /* Same compact-B preparation as the original OpenMP dispatch. */
    for(long k=0;k<t->k;k++) for(long j=0;j<t->n;j++)
        w->compact_b[k*t->n+j]=t->b[k*t->full_n+t->n0+j];
    for(long i=0;i<w->mb;i++)
        pack_a_panel(IME_PROFILE_A60,t->a,i*IME_MR,t->m,w->km,t->ap+i*w->as);
    for(long j=0;j<w->nb;j++)
        pack_b_panel(IME_PROFILE_A60,w->compact_b,j*IME_NR,t->n,w->km,t->bp+j*w->bs);
    return 0;
}
static int execute(BenchTile *t,BenchPhases *p,int profile) {
    ImeWorkspace *w=t->private_data;
    int32_t *c=t->c+t->n0*t->m;
    double s;
    /* Retain original tile-by-tile compute -> scatter -> K-tail order. */
    for(long nb=0;nb<w->nb;nb++) {
        long n0=nb*IME_NR;
        for(long mb=0;mb<w->mb;mb++) {
            long m0=mb*IME_MR;
            int32_t out[IME_MAX_OUTPUT_VALUES] __attribute__((aligned(64)));
            if(w->km) {
                s=profile?bench_now():0;
                native_accumulate(IME_PROFILE_A60,t->ap+mb*w->as,
                                  t->bp+nb*w->bs,w->km/(8*UNROLL_K),out);
                if(profile)p->kernel+=bench_now()-s;
                s=profile?bench_now():0;
                scatter_output(IME_PROFILE_A60,m0,n0,1,out,c,t->m);
                if(profile)p->output+=bench_now()-s;
            }
            if(w->km!=t->k) {
                s=profile?bench_now():0;
                scalar_gemm_block(IME_MR,IME_NR,t->k-w->km,1,
                    t->a+w->km*t->m+m0,t->m,w->compact_b+w->km*t->n+n0,t->n,
                    c+n0*t->m+m0,t->m);
                if(profile)p->boundary+=bench_now()-s;
            }
        }
        if(w->mb*IME_MR!=t->m) {
            s=profile?bench_now():0;
            scalar_gemm_block(t->m-w->mb*IME_MR,IME_NR,t->k,1,
                t->a+w->mb*IME_MR,t->m,w->compact_b+n0,t->n,
                c+n0*t->m+w->mb*IME_MR,t->m);
            if(profile)p->boundary+=bench_now()-s;
        }
    }
    if(w->nb*IME_NR!=t->n) {
        s=profile?bench_now():0;
        scalar_gemm_block(t->m,t->n-w->nb*IME_NR,t->k,1,t->a,t->m,
            w->compact_b+w->nb*IME_NR,t->n,c+w->nb*IME_NR*t->m,t->m);
        if(profile)p->boundary+=bench_now()-s;
    }
    return 0;
}
static void release(BenchTile *t) {
    ImeWorkspace *w=t->private_data;
    if(w) free(w->compact_b);
    free(w);free(t->ap);free(t->bp);
}
const BenchBackend ime_backend={"ime",0,0,supported,allocate,pack,execute,release};
