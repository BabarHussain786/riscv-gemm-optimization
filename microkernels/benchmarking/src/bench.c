#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sched.h>
#include <signal.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "counters.h"
#define MAX_WORKERS 256
typedef struct {
 long m,n,k; int threads,cpus[MAX_WORKERS],ime_workers,weight,chunk,warmups,reps;
 unsigned seed; int profile,counters,validate_only,skip_boundary_validation,prepacked,dynamic;
 const char *impl,*timing,*schedule;
} Options;
typedef struct {
 BenchPhases phases;BenchCounters counters;int cpu_before,cpu_after,failed;
 long strips;
} Worker;
typedef struct {
 long m,n,k,count;int8_t*a,*b;int32_t*c,*initial,*reference;
 BenchTile *rt,*it;
} Problem;
double bench_now(void){
#ifdef _WIN32
 LARGE_INTEGER v,f;QueryPerformanceCounter(&v);QueryPerformanceFrequency(&f);
 return (double)v.QuadPart/f.QuadPart;
#else
 struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;
#endif
}
void *bench_alloc(size_t n,size_t size){
 if(size && n>SIZE_MAX/size)return NULL;
 /* calloc guarantees initialized tails; kernel loads impose no alignment
  * requirement on these byte-packed panels. IME output tiles remain aligned. */
 return calloc(n?n:1,size);
}
static int cpu_now(void){
#ifdef _WIN32
 return (int)GetCurrentProcessorNumber();
#else
 return sched_getcpu();
#endif
}
#if !defined(_WIN32)
static void sigill(int sig,siginfo_t*info,void*context){
 (void)context;
 /* async-signal-safe address diagnostic, independent of stdio buffering. */
 const char prefix[]="IME/RVV SIGILL fault address=0x";
 char out[2*sizeof(uintptr_t)+2];uintptr_t x=(uintptr_t)info->si_addr;
 for(size_t i=0;i<2*sizeof(x);i++)out[i]="0123456789abcdef"[(x>>(4*(2*sizeof(x)-1-i)))&15];
 out[2*sizeof(x)]='\n';
 (void)write(2,prefix,sizeof(prefix)-1);(void)write(2,out,2*sizeof(x)+1);
 _exit(128+sig);
}
#endif
static int pin(int cpu){
#ifdef BENCH_HOST_TEST
 (void)cpu;return 0;
#elif defined(__linux__)
 cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);
 return sched_setaffinity(0,sizeof(mask),&mask)!=0 || sched_getcpu()!=cpu;
#else
 (void)cpu;return 1;
#endif
}
static unsigned next_random(unsigned*s){*s^=*s<<13;*s^=*s>>17;*s^=*s<<5;return *s;}
static void cleanup(Problem*p){
 if(p->rt)for(long i=0;i<p->count;i++)rvv_backend.release(&p->rt[i]);
 if(p->it)for(long i=0;i<p->count;i++)ime_backend.release(&p->it[i]);
 free(p->rt);free(p->it);free(p->a);free(p->b);free(p->c);free(p->initial);free(p->reference);
 memset(p,0,sizeof(*p));
}
static int prepare(Problem*p,Options*o,long m,long n,long k){
 memset(p,0,sizeof(*p));p->m=m;p->n=n;p->k=k;p->count=(n+31)/32;
 /* Reject overflow and excessively large test allocations before multiplying. */
 if(m<=0||n<=0||k<=0||m>INT_MAX||n>INT_MAX||k>INT_MAX ||
    (size_t)m>SIZE_MAX/(size_t)k || (size_t)n>SIZE_MAX/(size_t)k ||
    (size_t)m>SIZE_MAX/(size_t)n)return 1;
 size_t ac=(size_t)m*k,bc=(size_t)n*k,cc=(size_t)m*n;
 p->a=bench_alloc(ac,1);p->b=bench_alloc(bc,1);
 p->c=bench_alloc(cc,4);p->initial=bench_alloc(cc,4);p->reference=bench_alloc(cc,4);
 if(!p->a||!p->b||!p->c||!p->initial||!p->reference)goto failure;
 unsigned seed=o->seed?o->seed:1;
 /* Same canonical, bounded signed inputs for EVERY backend/configuration. */
 for(size_t i=0;i<ac;i++)p->a[i]=(int8_t)((int)(next_random(&seed)%15)-7);
 for(size_t i=0;i<bc;i++)p->b[i]=(int8_t)((int)(next_random(&seed)%15)-7);
 for(size_t i=0;i<cc;i++)p->initial[i]=(int32_t)((int)(next_random(&seed)%7)-3);
 /* Independent canonical INT64 reference; reject overflow, do not wrap. */
 for(long j=0;j<n;j++)for(long i=0;i<m;i++){
  int64_t sum=p->initial[j*m+i];
  for(long l=0;l<k;l++)sum+=(int64_t)p->a[l*m+i]*p->b[l*n+j];
  if(sum<INT32_MIN||sum>INT32_MAX){fprintf(stderr,"reference overflow\n");goto failure;}
  p->reference[j*m+i]=(int32_t)sum;
 }
 if(o->ime_workers<o->threads){p->rt=bench_alloc(p->count,sizeof(BenchTile));if(!p->rt)goto failure;}
 if(o->ime_workers){p->it=bench_alloc(p->count,sizeof(BenchTile));if(!p->it)goto failure;}
 for(long s=0;s<p->count;s++)for(int b=0;b<2;b++){
  BenchTile *array=b?p->it:p->rt;if(!array)continue;
  BenchTile*t=&array[s];t->m=m;t->n=n-s*32<32?n-s*32:32;t->k=k;
  t->full_n=n;t->n0=s*32;t->a=p->a;t->b=p->b;t->c=p->c;
  if((b?ime_backend:rvv_backend).allocate(t))goto failure;
 }
 return 0;
failure:cleanup(p);return 1;
}
static int validate_output(Problem*p){
 size_t mismatches=0;for(size_t i=0;i<(size_t)p->m*p->n;i++)if(p->c[i]!=p->reference[i]){
  if(!mismatches)fprintf(stderr,"first mismatch index=%zu actual=%d reference=%d\n",i,p->c[i],p->reference[i]);
  mismatches++;
 }
 if(mismatches)fprintf(stderr,"mismatches=%zu\n",mismatches);
 return mismatches!=0;
}
static void do_strip(Problem*p,Options*o,Worker*w,int ime,long s){
 const BenchBackend*b=ime?&ime_backend:&rvv_backend;
 BenchTile*t=ime?&p->it[s]:&p->rt[s];
 if(!o->prepacked){double start=o->profile?bench_now():0;
  if(b->pack(t))w->failed=1;
  if(o->profile)w->phases.packing+=bench_now()-start;
 }
 if(!w->failed && b->execute(t,&w->phases,o->profile))w->failed=1;
 w->strips++;
}
static int execute(Problem*p,Options*o,Worker*w,double*elapsed){
 memset(w,0,sizeof(*w)*o->threads);
 memcpy(p->c,p->initial,(size_t)p->m*p->n*sizeof(int32_t));
 /* Both layouts prepared when dynamic mixed ownership is unknown. These
  * preparations are outside the PREPACKED timer, never counted as E2E work. */
 if(o->prepacked)for(long s=0;s<p->count;s++){
  if(p->rt&&rvv_backend.pack(&p->rt[s]))return 1;
  if(p->it&&ime_backend.pack(&p->it[s]))return 1;
 }
 long queue=0;int failed=0;double start=0,finish=0;
#ifdef _OPENMP
 omp_set_dynamic(0);
#pragma omp parallel num_threads(o->threads) shared(queue,failed,start,finish)
#endif
 {
#ifdef _OPENMP
  int id=omp_get_thread_num();
  if(omp_get_num_threads()!=o->threads){
#pragma omp atomic write
   failed=1;
  }
#else
  int id=0;
#endif
  Worker*worker=&w[id];int ime=id<o->ime_workers;
  const BenchBackend*b=ime?&ime_backend:&rvv_backend;
  worker->failed=pin(o->cpus[id]);worker->cpu_before=cpu_now();
  if(!worker->failed && !b->supported()){
   fprintf(stderr,"unsupported backend=%s cpu=%d (K1 IME requires cpu-ai marker and A60 profile)\n",b->name,worker->cpu_before);
   worker->failed=1;
  }
  if(worker->failed){
#ifdef _OPENMP
#pragma omp atomic write
#endif
   failed=1;
  }
  counter_open(&worker->counters,o->counters);
#ifdef _OPENMP
#pragma omp barrier
#pragma omp master
#endif
  {start=bench_now();}
#ifdef _OPENMP
#pragma omp barrier
#endif
  /* Counters cover each worker's assignment+packing+compute+output region,
   * excluding team creation, affinity, reference, warmup and validation. */
  counter_start(&worker->counters);
  if(!failed){
   if(o->dynamic){
    for(;;){long first;
#ifdef _OPENMP
#pragma omp atomic capture
#endif
     {first=queue;queue+=o->chunk;}
     if(first>=p->count)break;
     for(long s=first;s<first+o->chunk&&s<p->count;s++)do_strip(p,o,worker,ime,s);
    }
   }else{
    long ime_count;
    if(!o->ime_workers)ime_count=0;
    else if(o->ime_workers==o->threads)ime_count=p->count;
    else ime_count=(p->count*o->weight+(o->weight+1)/2)/(o->weight+1);
    if(o->ime_workers && o->ime_workers<o->threads && p->count>1){
     if(ime_count==0)ime_count=1;if(ime_count==p->count)ime_count--;
    }
    long begin=ime?0:ime_count,end=ime?ime_count:p->count;
    int rank=ime?id:id-o->ime_workers;
    int team=ime?o->ime_workers:o->threads-o->ime_workers;
    for(long s=begin+rank;s<end;s+=team)do_strip(p,o,worker,ime,s);
   }
  }
  counter_stop(&worker->counters);worker->cpu_after=cpu_now();
#ifndef BENCH_HOST_TEST
  if(worker->cpu_before!=o->cpus[id]||worker->cpu_after!=o->cpus[id])worker->failed=1;
#endif
#ifdef _OPENMP
#pragma omp barrier
#pragma omp master
#endif
  {finish=bench_now();}
  counter_close(&worker->counters);
 }
 *elapsed=finish-start;
 long strips=0;for(int i=0;i<o->threads;i++){if(w[i].failed)failed=1;strips+=w[i].strips;}
 if(strips!=p->count)failed=1;
#ifdef BENCH_HOST_TEST
 if(getenv("BENCH_TEST_CORRUPT"))p->c[0]^=1;
#endif
 return failed||validate_output(p);
}
static void json_time(const char*name,double v,int available){
 printf(",\"%s\":",name);if(available)printf("%.17g",v);else printf("null");
}
static void emit(Problem*p,Options*o,Worker*w,double elapsed,int rep){
 BenchPhases total={0};uint64_t counts[4]={0};int counters_ok=o->counters;
 for(int i=0;i<o->threads;i++){
  total.packing+=w[i].phases.packing;total.kernel+=w[i].phases.kernel;
  total.output+=w[i].phases.output;total.boundary+=w[i].phases.boundary;
  counters_ok &=w[i].counters.ok;
  for(int j=0;j<4;j++)counts[j]+=w[i].counters.value[j];
 }
 time_t stamp=time(NULL);char utc[40];strftime(utc,sizeof(utc),"%Y-%m-%dT%H:%M:%SZ",gmtime(&stamp));
 printf("{\"timestamp_utc\":\"%s\",\"status\":\"OK\",\"validation\":\"PASS\",\"implementation\":\"%s\","
  "\"timing_mode\":\"%s\",\"datatype\":\"INT8_INT32\",\"M\":%ld,\"N\":%ld,\"K\":%ld,"
  "\"threads\":%d,\"rvv_workers\":%d,\"ime_workers\":%d,\"schedule\":\"%s\","
  "\"rep\":%d,\"seed\":%u,\"profiled\":%s,\"total_sec\":%.17g,\"gops\":%.17g,"
  "\"phase_aggregation\":\"sum_worker_elapsed\",\"rvv_output_fused\":%s,"
  "\"boundary_scope\":\"RVV_fused_IME_separate\",\"counter_scope\":\"sum_worker_measured_regions\","
  "\"counters_status\":\"%s\",\"cpu_ids\":[",
  utc,o->impl,o->timing,p->m,p->n,p->k,o->threads,o->threads-o->ime_workers,o->ime_workers,
  o->schedule,rep,o->seed,o->profile?"true":"false",elapsed,
  2.0*p->m*p->n*p->k/(elapsed*1e9),o->ime_workers<o->threads?"true":"false",
  counters_ok?"OK":(o->counters?"UNAVAILABLE_OR_MULTIPLEXED":"DISABLED"));
 for(int i=0;i<o->threads;i++)printf("%s%d",i?",":"",w[i].cpu_before);
 printf("]");
 json_time("packing_sec",total.packing,o->profile&&!o->prepacked);
 json_time("kernel_sec",total.kernel,o->profile);
 json_time("output_sec",total.output,o->profile&&o->ime_workers==o->threads);
 json_time("boundary_sec",total.boundary,o->profile&&o->ime_workers==o->threads);
 /* Mixed output fields identify IME contribution only, not total output cost. */
 json_time("ime_output_sec",total.output,o->profile&&o->ime_workers>0);
 json_time("ime_boundary_sec",total.boundary,o->profile&&o->ime_workers>0);
 const char*names[4]={"cycles","instructions","cache_references","cache_misses"};
 for(int j=0;j<4;j++){printf(",\"%s\":",names[j]);if(counters_ok)printf("%llu",(unsigned long long)counts[j]);else printf("null");}
 json_time("ipc",counts[0]?(double)counts[1]/counts[0]:0,counters_ok&&counts[0]>0);
 printf(",\"workers\":[");
 for(int i=0;i<o->threads;i++)printf("%s{\"id\":%d,\"cpu_before\":%d,\"cpu_after\":%d,\"strips\":%ld}",i?",":"",i,w[i].cpu_before,w[i].cpu_after,w[i].strips);
 printf("]}\n");fflush(stdout);
}
static long number(const char*s){char*end;errno=0;long n=strtol(s,&end,10);if(errno||*end){fprintf(stderr,"invalid integer: %s\n",s);exit(2);}return n;}
int main(int argc,char**argv){
 Options o={0};o.m=o.n=o.k=64;o.threads=1;o.weight=4;o.chunk=1;o.warmups=1;o.reps=6;o.seed=42;
 o.impl="rvv";o.timing="end_to_end";o.schedule="static";const char*cpus="0";
 for(int i=1;i<argc;i+=2){if(i+1>=argc){fprintf(stderr,"missing option value\n");return 2;}
  const char*k=argv[i],*v=argv[i+1];
  if(!strcmp(k,"--implementation"))o.impl=v;
  else if(!strcmp(k,"--timing"))o.timing=v;
  else if(!strcmp(k,"--schedule"))o.schedule=v;
  else if(!strcmp(k,"--cpus"))cpus=v;
  else if(!strcmp(k,"--m"))o.m=number(v);else if(!strcmp(k,"--n"))o.n=number(v);else if(!strcmp(k,"--k"))o.k=number(v);
  else if(!strcmp(k,"--threads"))o.threads=(int)number(v);
  else if(!strcmp(k,"--ime-workers"))o.ime_workers=(int)number(v);
  else if(!strcmp(k,"--weight"))o.weight=(int)number(v);
  else if(!strcmp(k,"--chunk"))o.chunk=(int)number(v);
  else if(!strcmp(k,"--warmups"))o.warmups=(int)number(v);
  else if(!strcmp(k,"--repetitions"))o.reps=(int)number(v);
  else if(!strcmp(k,"--seed"))o.seed=(unsigned)number(v);
  else if(!strcmp(k,"--profile"))o.profile=(int)number(v);
  else if(!strcmp(k,"--counters"))o.counters=(int)number(v);
  else if(!strcmp(k,"--validate-only"))o.validate_only=(int)number(v);
  else if(!strcmp(k,"--skip-boundary-validation"))o.skip_boundary_validation=(int)number(v);
  else{fprintf(stderr,"unknown option %s\n",k);return 2;}
 }
 if(o.threads<1||o.threads>MAX_WORKERS||o.ime_workers<0||o.ime_workers>o.threads||o.reps<1||o.warmups<0||o.weight<1||o.weight>1000||o.chunk<1)return 2;
 o.prepacked=!strcmp(o.timing,"prepacked");o.dynamic=!strcmp(o.schedule,"dynamic");
 if((!o.prepacked&&strcmp(o.timing,"end_to_end"))||(!o.dynamic&&strcmp(o.schedule,"static")))return 2;
 if(!strcmp(o.impl,"ime"))o.ime_workers=o.threads;
 else if(!strcmp(o.impl,"rvv")||!strcmp(o.impl,"reference")){if(o.ime_workers)return 2;}
 else if(strcmp(o.impl,"mixed")||o.ime_workers==0||o.ime_workers==o.threads)return 2;
#ifdef BENCH_HOST_TEST
 if(strcmp(o.impl,"reference")||o.threads!=1){fprintf(stderr,"HOST_TEST supports reference only, never RVV/IME\n");return 3;}
#else
 if(!strcmp(o.impl,"reference"))return 3;
#endif
#ifndef _OPENMP
 if(o.threads!=1){fprintf(stderr,"OpenMP required for multiple workers\n");return 3;}
#endif
 int ncpu=0;char*copy=malloc(strlen(cpus)+1);if(!copy)return 2;strcpy(copy,cpus);
 for(char*s=strtok(copy,",");s;s=strtok(NULL,",")){
  long c=number(s);if(c<0||c>=1024||ncpu>=MAX_WORKERS){free(copy);return 2;}
  for(int j=0;j<ncpu;j++)if(o.cpus[j]==c){free(copy);fprintf(stderr,"duplicate CPU / oversubscription\n");return 2;}
  o.cpus[ncpu++]=(int)c;
 }free(copy);if(ncpu!=o.threads){fprintf(stderr,"CPU count must equal workers\n");return 2;}
#if !defined(_WIN32)
 struct sigaction sa;memset(&sa,0,sizeof(sa));sa.sa_sigaction=sigill;sa.sa_flags=SA_SIGINFO;sigemptyset(&sa.sa_mask);sigaction(SIGILL,&sa,NULL);
#endif
#if defined(__linux__) && !defined(BENCH_HOST_TEST)
 cpu_set_t allowed;if(sched_getaffinity(0,sizeof(allowed),&allowed))return 3;
 for(int i=0;i<o.threads;i++)if(!CPU_ISSET(o.cpus[i],&allowed)){fprintf(stderr,"CPU %d outside permitted affinity\n",o.cpus[i]);return 3;}
#endif
 Worker*w=calloc(o.threads,sizeof(*w));if(!w)return 2;
 long shapes[3][3]={{16,16,64},{15,15,69},{o.m,o.n,o.k}};
 /* Tiny mixed strips may be assigned entirely to one backend. Explicitly
  * validate both worker kernels on complete and boundary shapes as well. */
 if(o.ime_workers>0 && o.ime_workers<o.threads){
  int backend_shape_count=o.skip_boundary_validation?1:2;
  for(int backend=0;backend<2;backend++)for(int shape=0;shape<backend_shape_count;shape++){
   Options single=o;single.threads=1;single.ime_workers=backend;
   single.cpus[0]=o.cpus[backend?0:o.ime_workers];single.dynamic=0;
   single.schedule="static";single.impl=backend?"ime":"rvv";
   Problem p;double elapsed;
   if(prepare(&p,&single,shapes[shape][0],shapes[shape][1],shapes[shape][2])){free(w);return 4;}
   int rc=execute(&p,&single,w,&elapsed);
   fprintf(stderr,"BACKEND_VALIDATION backend=%s shape=%ldx%ldx%ld status=%s\n",single.impl,p.m,p.n,p.k,rc?"FAILED":"PASS");
   cleanup(&p);if(rc){free(w);return 5;}
  }
 }
 /* Fig. 5 measures the requested full workload.  Its optional boundary
  * check is kept separate so a tail-shape fault cannot suppress 1024^3
  * timing data; the dedicated correctness campaign still exercises it. */
 int validation_shapes[2]={0,2};
 int validation_shape_count=o.skip_boundary_validation?2:3;
 for(int index=0;index<validation_shape_count;index++){
  int shape=o.skip_boundary_validation?validation_shapes[index]:index;
  Problem p;double elapsed;
  if(prepare(&p,&o,shapes[shape][0],shapes[shape][1],shapes[shape][2])){free(w);return 4;}
  int rc=execute(&p,&o,w,&elapsed);
  fprintf(stderr,"VALIDATION shape=%ldx%ldx%ld status=%s\n",p.m,p.n,p.k,rc?"FAILED":"PASS");
  cleanup(&p);if(rc){free(w);return 5;}
 }
 if(o.validate_only){printf("{\"record_type\":\"validation\",\"status\":\"OK\",\"validation\":\"PASS\"}\n");free(w);return 0;}
 Problem p;if(prepare(&p,&o,o.m,o.n,o.k)){free(w);return 4;}
 for(int rep=1-o.warmups;rep<=o.reps;rep++){
  double elapsed;if(execute(&p,&o,w,&elapsed)||elapsed<=0){cleanup(&p);free(w);return 5;}
  if(rep>0)emit(&p,&o,w,elapsed,rep);
 }
 cleanup(&p);free(w);return 0;
}
