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
#ifndef CPU_IPC_CU
#define CPU_IPC_CU
#include <assert.h>
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "fs_constants.h"
#include "fs_globals.cu.h"
__host__ void CPU_IPC_RW_Queue::init_host() volatile
{
	for (int i = 0; i < RW_IPC_SIZE; i++)
	{
		entries[i].init_host();
		push_list[i] = 0;// complete_list[i] = 0;
	}
}
__host__ void CPU_IPC_RW_Entry::init_host() volatile
{
	key = 0;
	offset = 0;
	size = 0;
	mode = 0;
	return_value = 0;
}
__device__ void GPU_IPC_RW_Manager::init_thread() volatile
{
	for (int i = 0; i < RW_IPC_SIZE; i++)
	{
		_locker[i] = IPC_MGR_EMPTY;
	}
	for(int i=0;i<MAX_BLOCK;i++){
		_block_lock[i]=IPC_MGR_EMPTY;
	}
	_lock = 0;
}
__device__ void CPU_IPC_RW_Entry::request(unsigned long _key,unsigned long _offset,
	unsigned int _size, unsigned int _mode) volatile
{
	key = _key;
	offset = _offset;
	size = _size;
	mode = _mode;
}


__device__ int GPU_IPC_RW_Manager::findEntry(int mode) volatile
{
	//const int init = (blockIdx.x + threadIdx.y) & (RW_IPC_SIZE - 1); //KVKVKVKVKV need change!
	// TODO -> lockfree optimization, just attempt to take private TB lock trivial and will work well!!
	// assuming one concurrent call PER TB 
	int i,init,num=1;
	if (mode == MODE_BLOCK) init = BLOCK_ID & (RW_IPC_SIZE - 1); //block
	else if(mode==MODE_WARP) init = (BLOCK_ID*NUM_WARPS+WARP_ID) & (RW_IPC_SIZE - 1); //warp
	else {
		init = (BLOCK_ID*NUM_THREADS+(WARP_ID*32)) & (RW_IPC_SIZE - 1); //thread 
		num = 32;
	}
	i = init; 

	do
	{
        //thread단위에서 하나만 lock해도, 32 배수 단위로 접근하고, size도 32배수로 할꺼라 상관없음!
		if (atomicExch((int*)&_locker[i], IPC_MGR_BUSY) == IPC_MGR_EMPTY)
		{
			return i;
		}
		i = (i + num) & (RW_IPC_SIZE - 1);
	} while (1);
}

__device__ void GPU_IPC_RW_Manager::freeEntry(int entry) volatile
{
	assert(_locker[entry] == IPC_MGR_BUSY);
	_locker[entry] = IPC_MGR_EMPTY;
	//__threadfence();
}
__device__ int CPU_IPC_RW_Queue::request_thread(unsigned long key,
	unsigned long offset, unsigned int size,unsigned int mode) volatile
{
	int entry;
	int return_value;
	int laneid = LANE_ID;//threadIdx.x & 0x1f;
	
	if( laneid == 0)
	{
		entry = g_ipcRWManager->findEntry(MODE_THREAD);
		REQUEST_ADD_MUL(32)
	}
	entry = broadcast(entry);
	entry = entry + laneid;
	
	CPU_READ_START_WARP

		entries[entry].request(key, offset, size, mode); //all threads request!

	__threadfence_system();

	GPU_ASSERT(push_list[entry] == CPU_IPC_EMPTY);
	push_list[entry] = CPU_IPC_PENDING;

	__threadfence_system();

	if (laneid == 0) {
		int i;
		for (i = 0; i < 32; i++) {
			if(push_list[entry+i] != CPU_IPC_EMPTY)
				WAIT_ON_MEM(push_list[entry + i], CPU_IPC_COMPLETE);
		}
	}

	CPU_READ_STOP_WARP
	
	return_value = readNoCache(&entries[entry].return_value);
	push_list[entry] = CPU_IPC_EMPTY;

	__threadfence_system();
	/////////__threadfence_system(); //maybe????
	
	if (laneid == 0) {
		g_ipcRWManager->freeEntry(entry);
	}
	return return_value;
}
__device__ int CPU_IPC_RW_Queue::request_thread_bad1(unsigned long key,
	unsigned long offset, unsigned int size,unsigned int mode) volatile 
{//모두가 memory read
	int entry;
	int return_value;
	int laneid = LANE_ID;//threadIdx.x & 0x1f;
	
	if (laneid == 0)
	{
		entry = g_ipcRWManager->findEntry(MODE_THREAD);
		REQUEST_ADD_MUL(32)
	}
	entry = broadcast(entry);
	entry = entry + laneid;
	
	CPU_READ_START_WARP

		entries[entry].request(key, offset, size, mode); //all threads request!

	__threadfence_system();

	GPU_ASSERT(push_list[entry] == CPU_IPC_EMPTY);
	push_list[entry] = CPU_IPC_PENDING;

	__threadfence_system();

	WAIT_ON_MEM(push_list[entry ], CPU_IPC_COMPLETE);
	
	CPU_READ_STOP_WARP
	
	return_value = readNoCache(&entries[entry].return_value);
	push_list[entry] = CPU_IPC_EMPTY;

	__threadfence_system();
	/////////__threadfence_system(); //maybe????
	
	if (laneid == 0) {
		g_ipcRWManager->freeEntry(entry);
	}
	return return_value;
}


__device__ int CPU_IPC_RW_Queue::request_warp(unsigned long key,
	unsigned long offset, unsigned int size,unsigned int mode) volatile
{
	int entry,return_value;
	entry = g_ipcRWManager->findEntry(MODE_WARP);
	
	CPU_READ_START_WARP

		entries[entry].request(key, offset, size, mode);

	__threadfence_system();
	GPU_ASSERT(push_list[entry] == CPU_IPC_EMPTY);
	push_list[entry] = CPU_IPC_PENDING;

	__threadfence_system();

	WAIT_ON_MEM(push_list[entry], CPU_IPC_COMPLETE);

	CPU_READ_STOP_WARP

	return_value = readNoCache(&entries[entry].return_value);
	push_list[entry] = CPU_IPC_EMPTY;
	
	__threadfence_system();
	/////////__threadfence_system(); //maybe????
	g_ipcRWManager->freeEntry(entry);
	return return_value;
}

__device__ int CPU_IPC_RW_Queue::request_block(unsigned long key,
	unsigned long offset, unsigned int size, unsigned int mode) volatile
{
	int entry,return_value;
	entry = g_ipcRWManager->findEntry(MODE_BLOCK);

	CPU_READ_START_WARP

		entries[entry].request(key, offset, size, mode);

	__threadfence_system();
	GPU_ASSERT(push_list[entry] == CPU_IPC_EMPTY);
	push_list[entry] = CPU_IPC_PENDING;

	__threadfence_system();

	WAIT_ON_MEM(push_list[entry], CPU_IPC_COMPLETE);

	CPU_READ_STOP_WARP

		//push_list[entry] = CPU_IPC_EMPTY; ////in cpu-side, really clean?
	return_value = readNoCache(&entries[entry].return_value);
	push_list[entry] = CPU_IPC_EMPTY;
	__threadfence_system();
	/////////__threadfence_system(); //maybe????
	g_ipcRWManager->freeEntry(entry);
	return return_value;
}


//////////////////////////////////////////////////////////////

#endif



