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

#ifndef FS_CALLS_CU
#define FS_CALLS_CU

#include "fs_calls.cu.h"
 // no reference counting here

 // we must have multiple threads otherwise it 

DEBUG_NOINLINE __device__ int key_value_op_thread(unsigned long key, unsigned long offset,
	unsigned int size, unsigned int mode)
{
	int return_value;

	LATENCY_READ_START_WARP
	
	return_value = g_cpu_ipcRWQueue->request_thread(key, offset, size, mode);

	LATENCY_READ_STOP_WARP
	
	return return_value;
}

DEBUG_NOINLINE __device__ int key_value_op_thread_bad1(unsigned long key, unsigned long offset,
	unsigned int size, unsigned int mode)
{
	int return_value;

	LATENCY_READ_START_WARP

	return_value = g_cpu_ipcRWQueue->request_thread_bad1(key, offset, size, mode);

	LATENCY_READ_STOP_WARP
	
	return return_value;
}

DEBUG_NOINLINE __device__ int key_value_op_warp(unsigned long key, unsigned long offset,
	unsigned int size, unsigned int mode) 
{
//	int laneid = threadIdx.x & 0x1f;
	int return_value;
	LATENCY_READ_START_WARP

	if (LANE_ID == 0) 
	{
		return_value = g_cpu_ipcRWQueue->request_warp(key, offset, size, mode);
		
		REQUEST_ADD
	}
	return_value = broadcast(return_value);

	LATENCY_READ_STOP_WARP

	return return_value;
}
DEBUG_NOINLINE __device__ int key_value_op_block(unsigned long key, unsigned long offset,
	unsigned int size, unsigned int mode) //가장 처음 도착하는 것이 request 요청
{
	__shared__ int return_value,winner;
	
	//int laneid=threadIdx.x & 0x1f;
	
//	LATENCY_READ_START_WARP
	
	if(LANE_ID==0 && atomicExch((int*)&(g_ipcRWManager->_block_lock[BLOCK_ID]),IPC_MGR_BUSY)==IPC_MGR_EMPTY){
//		REQUEST_ADD
			
		winner=TID;
		//printf("tid: %d, request %lu, offset: %lu\n",winner, key, offset);
		return_value = g_cpu_ipcRWQueue->request_block(key, offset, size, mode);

	END_BLOCK_SINGLE_THREAD

	if(winner==TID){
		g_ipcRWManager->_block_lock[BLOCK_ID]=IPC_MGR_EMPTY;
		
//		LATENCY_READ_STOP_WARP ///winner 시간만 더함.

	}

	return return_value;
}
DEBUG_NOINLINE __device__ int key_value_op_block_bad1(unsigned long key, unsigned long offset,
	unsigned int size, unsigned int mode)//block의 첫 쓰레드가 request 처리 request 시작시에는 no sync
{
	__shared__ int return_value,winner;


	LATENCY_READ_START_WARP
	if(FOR_LATENCY){
		if(LANE_ID==0 && atomicExch((int*)&(g_ipcRWManager->_block_lock[BLOCK_ID]),IPC_MGR_BUSY)==IPC_MGR_EMPTY){
			winner=TID;
		}
	}

	BEGIN_BLOCK_SINGLE_THREAD
	
	REQUEST_ADD
	
	return_value = g_cpu_ipcRWQueue->request_block(key, offset, size, mode);
	
	END_BLOCK_SINGLE_THREAD

	if(FOR_LATENCY){
		if(winner==TID){
			g_ipcRWManager->_block_lock[BLOCK_ID]=IPC_MGR_EMPTY;
			LATENCY_READ_STOP_WARP
		}
	}

	return return_value;
}
DEBUG_NOINLINE __device__ int key_value_op_block_bad2(unsigned long key, unsigned long offset,
	unsigned int size, unsigned int mode) //처음과 끝 모두 sync
{
	__shared__ int return_value,winner;

	LATENCY_READ_START_WARP
	if(FOR_LATENCY){
		if(LANE_ID==0 && atomicExch((int*)&(g_ipcRWManager->_block_lock[BLOCK_ID]),IPC_MGR_BUSY)==IPC_MGR_EMPTY){
			winner=TID;
		}
	}

	BEGIN_SINGLE_THREAD
	
	REQUEST_ADD

	return_value = g_cpu_ipcRWQueue->request_block(key, offset, size, mode);
	
	END_SINGLE_THREAD

	if(FOR_LATENCY){
		if(winner==TID){
			g_ipcRWManager->_block_lock[BLOCK_ID]=IPC_MGR_EMPTY;
			LATENCY_READ_STOP_WARP
		}
	}

	return return_value;
}

#endif

////////////////////////////////////////////////////////////////
