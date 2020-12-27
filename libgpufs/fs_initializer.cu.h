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
#ifndef FS_INITIALIZER_H
#define  FS_INITIALIZER_H


//#ifndef MAIN_FS_FILE
//#error "This file must be included in the fs.cu"
//#endif

#include <stdio.h>

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "timer.h"
#include "fs_globals.cu.h"

#define  initGpuShmemPtr(T, h_ptr,symbol)\
{\
 	CUDA_SAFE_CALL(cudaHostAlloc((void**)&(h_ptr), sizeof(T), cudaHostAllocMapped));\
	memset((void*)h_ptr,0,sizeof(T));\
	void* d_ptr;\
	CUDA_SAFE_CALL(cudaHostGetDevicePointer((void**)(&d_ptr), (void*)(h_ptr), 0));\
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((symbol),&d_ptr,sizeof(void*)));\
}

#define initGpuGlobals(T,d_ptr,symbol)\
{\
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_ptr),sizeof(T)));\
	CUDA_SAFE_CALL(cudaMemset((void*)d_ptr,0,sizeof(T)));\
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((symbol),&(d_ptr),sizeof(void*)));\
}

struct GPUStreamManager
{
	// GPU streams 	
	int work;
	GPUStreamManager(int worker) {
		work=worker;
		CUDA_SAFE_CALL(cudaStreamCreate(&kernelStream));
		memStream=(cudaStream_t*)malloc(sizeof(cudaStream_t)*worker);
		for(int i=0;i<worker;i++){
			CUDA_SAFE_CALL(cudaStreamCreate(&memStream[i]));
		}

	}

	~GPUStreamManager() {
		CUDA_SAFE_CALL(cudaStreamDestroy(kernelStream));
		for(int i=0;i<work;i++){
			CUDA_SAFE_CALL(cudaStreamDestroy(memStream[i]));
		}
		free(memStream);
	}

	cudaStream_t kernelStream;
	cudaStream_t *memStream;
};

struct GPUGlobals;

struct TaskData
{
	int gpuid;
	int id;
	volatile GPUGlobals* gpuGlobals;
};

struct GPUGlobals {

	volatile CPU_IPC_RW_Queue* cpu_ipcRWQueue;

	// RW GPU manager
	GPU_IPC_RW_Manager* ipcRWManager;
	
	// Streams
	GPUStreamManager* streamMgr;
	char *user_put,*cuda_put;

	GPUGlobals(int worker)
	{
		// T / h_ptr / symbol
		initGpuShmemPtr(CPU_IPC_RW_Queue, cpu_ipcRWQueue, g_cpu_ipcRWQueue);
		cpu_ipcRWQueue->init_host();

		initGpuGlobals(GPU_IPC_RW_Manager, ipcRWManager, g_ipcRWManager);
		
		
		streamMgr = new GPUStreamManager(worker);

	}

	~GPUGlobals()
	{
		cudaFreeHost((void*)cpu_ipcRWQueue);

		cudaFree(ipcRWManager);
		delete streamMgr;
	}
};
typedef volatile GPUGlobals* GPUGlobals_ptr;

void initializer(GPUGlobals_ptr* globals,int worker);
#endif // FS_INITIALIZER_H

