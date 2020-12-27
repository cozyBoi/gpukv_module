/* 
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */

/* 
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */

#ifndef FS_DEBUG_CU_H
#define FS_DEBUG_CU_H

#include <stdint.h>
#include <stdio.h>

//#define DEBUG   ///////adddddd kv
//#define MALLOC_STATS
//#define TIMING_STATS



#ifdef DEBUG

struct gdebug_t;
extern            volatile struct gdebug_t *_hdbg;
extern __device__ volatile struct gdebug_t *_gdbg;
extern __device__ volatile int             _gdbg_mutex;

__host__ void _gdebug_init(void);

__device__ void _dbg(const char *s,
                       void* ptr,
                       size_t v,
                       long int t,
                       long int l,
                       const char *fname,
                       const char *func);

#define DBGT(s, ptr, v, t) _dbg(s, (void*)ptr, (size_t)v, t, __LINE__, __FILE__, __func__)
#define DBG_INIT() _gdebug_init()

#else

#define DBGT(s, ptr, v, t)
#define DBG_INIT()

#endif

#define GDBG(s, ptr, v) DBGT(s, (void*)ptr, (size_t)v, 0)
#define GDBGS(s) GDBG(s, (void*)-1, (size_t)-1)
#define GDBGL() GDBG("", (void*)-1, (size_t)-1)
#define GDBGV(s, v) GDBG(s, (void*)-1, (size_t)v)

#if DEBUG
#define GPRINT(...) \
	if( (threadIdx.x + threadIdx.y + threadIdx.z) ==0 ) \
	{ \
		printf(__VA_ARGS__); \
	}
#else
#define GPRINT(...)
#endif

#define PRINT_STATS(SYMBOL) { unsigned int tmp;\
			     cudaMemcpyFromSymbol(&tmp,SYMBOL,sizeof(int),0,cudaMemcpyDeviceToHost);\
			     fprintf(stderr,"%s %u\n", #SYMBOL, tmp);}

#define INIT_STATS(SYMBOL) SYMBOL=0;

/*** malloc stats****/
#ifdef MALLOC_STATS

extern __device__ unsigned int numRequest;

#define INIT_COUNT_STAT INIT_STATS(numRequest);
#define REQUEST_ADD atomicAdd(&numRequest,1);
#define REQUEST_ADD_MUL(num) atomicAdd(&numRequest,num);

#define PRINT_REQUEST PRINT_STATS(numRequest);


#else
#define INIT_COUNT_STAT
#define REQUEST_ADD
#define REQUEST_ADD_MUL(num)
#define PRINT_REQUEST

#endif

/***timing stats****/
#ifdef TIMING_STATS

extern __device__ unsigned long long KernelTime;

extern __device__ unsigned long long CPUReadTime;
extern __device__ unsigned long long LatencyTime;


#define PRINT_TIME(SYMBOL, blocks) { unsigned long long tmp; \
			     cudaMemcpyFromSymbol(&tmp,SYMBOL,sizeof(unsigned long long),0,cudaMemcpyDeviceToHost); \
			     printf("%s %fms\n", #SYMBOL, ((double)(tmp) / 1e6) / (double)(blocks)); }

#define GET_TIME(timer) \
	asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(timer) :);

#define START(timer) \
	unsigned long long timer##Start; \
	if( TID == 0 ) \
	{ \
		GET_TIME( timer##Start ); \
	}

#define START_WARP(timer) \
	unsigned long long timer##Start; \
	if( LANE_ID == 0 ) \
	{ \
		GET_TIME( timer##Start ); \
	}

#define STOP(timer) \
	unsigned long long timer##Stop; \
	if( TID == 0 ) \
	{ \
		GET_TIME( timer##Stop ); \
		atomicAdd(&timer##Time, timer##Stop - timer##Start); \
	}

#define STOP_WARP(timer) \
	unsigned long long timer##Stop; \
	if( LANE_ID == 0 ) \
	{ \
		GET_TIME( timer##Stop ); \
		atomicAdd(&timer##Time, timer##Stop - timer##Start); \
	}

#define INIT_RT_TIMING \
	INIT_STATS(KernelTime); \
	INIT_STATS(CPUReadTime); \
	INIT_STATS(LatencyTime);


//////////////////////////////
// THreadblock level timers //
//////////////////////////////
#define KERNEL_START START( Kernel )
#define KERNEL_STOP STOP( Kernel )

#define CPU_READ_START START( CPURead )
#define CPU_READ_STOP STOP( CPURead )

#define LATENCY_READ_START START( Latency )
#define LATENCY_READ_STOP STOP( Latency )

#define FOR_LATENCY 1
///////////////////////
// Warp level timers //
///////////////////////
#define KERNEL_START_WARP START_WARP( Kernel )
#define KERNEL_STOP_WARP STOP_WARP( Kernel )

#define CPU_READ_START_WARP START_WARP( CPURead )
#define CPU_READ_STOP_WARP STOP_WARP( CPURead )

#define LATENCY_READ_START_WARP START_WARP( Latency )
#define LATENCY_READ_STOP_WARP STOP_WARP( Latency )

//////////////////
// Timers print //
//////////////////
#define PRINT_KERNEL_TIME(blocks) PRINT_TIME(KernelTime, blocks);

#define PRINT_CPU_READ_TIME(blocks) PRINT_TIME(CPUReadTime, blocks);

#define PRINT_LATENCY_READ_TIME(blocks) PRINT_TIME(LatencyTime, blocks);


#else

#define PRINT_TIME(SYMBOL, blocks)

#define GET_TIME(timer)

#define INIT_RT_TIMING

//////////////////////////////
// THreadblock level timers //
//////////////////////////////
#define KERNEL_START
#define KERNEL_STOP

#define CPU_READ_START
#define CPU_READ_STOP

#define LATENCY_READ_START
#define LATENCY_READ_STOP
///////////////////////
// Warp level timers //
///////////////////////
#define KERNEL_START_WARP
#define KERNEL_STOP_WARP

#define CPU_READ_START_WARP
#define CPU_READ_STOP_WARP

#define LATENCY_READ_START_WARP
#define LATENCY_READ_STOP_WARP

#define FOR_LATENCY 0
//////////////////
// Timers print //
//////////////////
#define PRINT_KERNEL_TIME(blocks)

#define PRINT_CPU_READ_TIME(blocks)

#define PRINT_LATENCY_READ_TIME(blocks)

#endif

#define INIT_ALL_STATS { INIT_COUNT_STAT; }
#define INIT_TIMING_STATS { INIT_RT_TIMING; }
#endif

