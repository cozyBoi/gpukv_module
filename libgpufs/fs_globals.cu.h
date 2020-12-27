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


#ifndef fs_globals_cu_h
#define fs_globals_cu_h
#include <assert.h>
#include "cpu_ipc.cu.h"

/************GLOBALS********/

// CPU Write-shared memory //
extern __device__ volatile CPU_IPC_RW_Queue* g_cpu_ipcRWQueue;

// manager for rw RPC queue
extern __device__ volatile GPU_IPC_RW_Manager* g_ipcRWManager;


#endif
