#ifndef BENCH_COUNTERS_H
#define BENCH_COUNTERS_H
#include <stdint.h>
#include <string.h>
typedef struct {int fd[4],ok; uint64_t value[4],enabled[4],running[4];} BenchCounters;
#if defined(__linux__) && !defined(BENCH_HOST_TEST)
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
static void counter_close(BenchCounters*c){for(int i=0;i<4;i++)if(c->fd[i]>=0)close(c->fd[i]);}
static void counter_open(BenchCounters*c,int requested){
 memset(c,0,sizeof(*c));for(int i=0;i<4;i++)c->fd[i]=-1;
 if(!requested)return;
 uint64_t config[4]={PERF_COUNT_HW_CPU_CYCLES,PERF_COUNT_HW_INSTRUCTIONS,
                    PERF_COUNT_HW_CACHE_REFERENCES,PERF_COUNT_HW_CACHE_MISSES};
 for(int i=0;i<4;i++){
  struct perf_event_attr a;memset(&a,0,sizeof(a));a.type=PERF_TYPE_HARDWARE;
  a.size=sizeof(a);a.config=config[i];a.disabled=1;a.exclude_kernel=1;a.exclude_hv=1;
  a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
  c->fd[i]=(int)syscall(__NR_perf_event_open,&a,0,-1,i?c->fd[0]:-1,0);
  if(c->fd[i]<0){counter_close(c);for(int j=0;j<4;j++)c->fd[j]=-1;return;}
 }c->ok=1;
}
static void counter_start(BenchCounters*c){
 if(c->ok && (ioctl(c->fd[0],PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP)<0 ||
             ioctl(c->fd[0],PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP)<0))c->ok=0;
}
static void counter_stop(BenchCounters*c){
 if(c->fd[0]<0)return;
 if(ioctl(c->fd[0],PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP)<0)c->ok=0;
 for(int i=0;i<4;i++){
  uint64_t v[3]={0};
  if(read(c->fd[i],v,sizeof(v))!=(ssize_t)sizeof(v)){c->ok=0;continue;}
  c->value[i]=v[0];c->enabled[i]=v[1];c->running[i]=v[2];
  /* Never interpret unavailable or multiplexed counts as exact measurements. */
  if(!v[1] || v[1]!=v[2])c->ok=0;
 }
}
#else
static void counter_open(BenchCounters*c,int requested){(void)requested;memset(c,0,sizeof(*c));}
static void counter_start(BenchCounters*c){(void)c;}
static void counter_stop(BenchCounters*c){(void)c;}
static void counter_close(BenchCounters*c){(void)c;}
#endif
#endif
