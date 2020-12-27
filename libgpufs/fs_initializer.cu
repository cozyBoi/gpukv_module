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




//#ifndef MAIN_FS_FILE
//#error "This file must be included in the fs.cu"
//#endif

#ifndef FS_INITIALIZER
#define FS_INITIALIZER

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "timer.h"
#include "fs_globals.cu.h"

#include "fs_initializer.cu.h"

#include <nvToolsExt.h>


/************GLOBALS********/
// CPU Write-shared memory //
__device__ volatile CPU_IPC_RW_Queue* g_cpu_ipcRWQueue;
//
// manager for rw RPC queue

__device__ volatile GPU_IPC_RW_Manager* g_ipcRWManager;


__global__ void init_fs(volatile CPU_IPC_RW_Queue* _ipcRWQueue,
	volatile GPU_IPC_RW_Manager* _ipcRWManager)
{
	g_cpu_ipcRWQueue = _ipcRWQueue;

	g_ipcRWManager = _ipcRWManager;
	g_ipcRWManager->init_thread();


	INIT_ALL_STATS
		INIT_TIMING_STATS
		//INIT_DEBUG
}

typedef volatile GPUGlobals* GPUGlobals_ptr;

void initializer(GPUGlobals_ptr* globals, int worker)
{
//	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	*globals = new GPUGlobals(worker);

	//ssd_init( (*globals)->stagingArea, sizeof(uchar) * RW_HOST_WORKERS * RW_SCRATCH_PER_WORKER * FS_BLOCKSIZE * RW_SLOTS_PER_WORKER );
	
	// this must be done from a single thread!
	init_fs << <1, 1 >> >(
		(*globals)->cpu_ipcRWQueue,
		(*globals)->ipcRWManager);

	cudaThreadSynchronize();
	CUDA_SAFE_CALL(cudaPeekAtLastError());

	//	pthread_attr_t attr;
	//	pthread_attr_init( &attr );
	//	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
	//
	//	(*globals)->done = 0;
	//
	//	for( int i = 0; i < RW_HOST_WORKERS; ++i )
	//	{
	//		(*globals)->rwLoopTasksData[i].id = i;
	//		(*globals)->rwLoopTasksData[i].gpuGlobals =  *globals;
	//		(*globals)->rwLoopTasksData[i].gpuid = 0;
	//
	//		pthread_create( (pthread_t*)&((*globals)->rwLoopTasksIDs[i]), &attr, rw_task, (TaskData*)&((*globals)->rwLoopTasksData[i]) );
	//	}
	//
	//	pthread_attr_destroy( &attr );
}
#endif

