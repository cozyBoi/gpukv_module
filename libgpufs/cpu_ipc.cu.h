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

#ifndef CPU_IPC_H
#define CPU_IPC_H

struct CPU_IPC_RW_Entry
{
	volatile unsigned long key;
	volatile unsigned long offset;
	volatile unsigned int size;
	volatile unsigned int mode; //2:get, 1:put
	///////////////////////////////////////////////////24byte_request
	volatile int return_value;		//4byte_result
	
	__device__ void clean() volatile;

	__device__ void request(unsigned long _key, unsigned long _offset, unsigned int _size, unsigned int _mode) volatile;

	__host__ void init_host() volatile;
};

struct GPU_IPC_RW_Manager
{
	volatile int _locker[RW_IPC_SIZE];
	volatile int _block_lock[MAX_BLOCK];
	int _lock;
	__device__ int findEntry(int mode) volatile;
	__device__ void freeEntry(int entry) volatile;
	__device__ void init_thread() volatile;
};

struct CPU_IPC_RW_Queue
{
	volatile CPU_IPC_RW_Entry entries[RW_IPC_SIZE];
	volatile char push_list[RW_IPC_SIZE];
	//volatile char complete_list[RW_IPC_SIZE];

	__host__ void init_host() volatile;
	__device__ int request_block(unsigned long key, unsigned long offset, unsigned int size,
		unsigned int mode) volatile;
	__device__ int request_warp(unsigned long key, unsigned long offset, unsigned int size,
		unsigned int mode) volatile;
	__device__ int request_thread(unsigned long key, unsigned long offset, unsigned int size,
		unsigned int mode) volatile;
	__device__ int request_thread_bad1(unsigned long key, unsigned long offset, unsigned int size,
		unsigned int mode) volatile;
};


//__device__ int request_cpu(unsigned long key, unsigned long offset, unsigned int size, unsigned int mode);

#endif
///////////////////////////////////////////////////
