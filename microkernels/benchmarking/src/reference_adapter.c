/* HOST TEST ONLY. This is not RVV/IME emulation or publishable performance. */
#include "bench.h"
#include <stdlib.h>
static int available(void){return 1;}
static int unavailable(void){return 0;}
static int allocate(BenchTile*t){
 t->ap=bench_alloc((size_t)t->m*t->k,1);t->bp=bench_alloc((size_t)t->n*t->k,1);
 return !t->ap||!t->bp;
}
static int pack(BenchTile*t){
 for(long k=0;k<t->k;k++){
  for(long i=0;i<t->m;i++)t->ap[k*t->m+i]=t->a[k*t->m+i];
  for(long j=0;j<t->n;j++)t->bp[k*t->n+j]=t->b[k*t->full_n+t->n0+j];
 }return 0;
}
static int execute(BenchTile*t,BenchPhases*p,int profile){
 double s=profile?bench_now():0;
 for(long j=0;j<t->n;j++)for(long i=0;i<t->m;i++){
  int64_t sum=t->c[(t->n0+j)*t->m+i];
  for(long k=0;k<t->k;k++)sum+=(int32_t)t->ap[k*t->m+i]*t->bp[k*t->n+j];
  t->c[(t->n0+j)*t->m+i]=(int32_t)sum;
 }
 if(profile)p->kernel+=bench_now()-s;
 return 0;
}
static void release(BenchTile*t){free(t->ap);free(t->bp);}
const BenchBackend rvv_backend={"reference",1,1,available,allocate,pack,execute,release};
const BenchBackend ime_backend={"unsupported",0,0,unavailable,allocate,pack,execute,release};
