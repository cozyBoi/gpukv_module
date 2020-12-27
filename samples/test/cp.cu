#include <stdio.h>
#include <errno.h>

#include "fs_calls.cu.h"
#include "host_loop.h"
#include "util.cu.h"

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(err)  __checkCudaErrors (err, __FILE__, __LINE__)
#endif

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int) err, cudaGetErrorString(err));
		exit(-1);
	}
}


__global__ void test_use()
{
	unsigned long id =  BLOCK_ID*NUM_WARPS+WARP_ID;
	unsigned long offset =id+4096;
	key_value_op_warp(id, offset, 4096, GET);
	id+=1000; offset+=10000;
	key_value_op_warp(id, offset, 4096, GET);
}

int main(int argc, char** argv)
{
	int device = 0;

	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));

	printf("Running on device %d: \"%s\"\n", device, deviceProp.name);
	/*
	if (argc < 5)
	{
		fprintf(stderr, "<kernel_iterations> <blocks> <threads> f1 f2 ... f_#files\n");
		return -1;
	}*/

	int nblocks = 28;
	int nthreads = 1024;

	fprintf(stderr, "\tblocks %d threads %d\n", nblocks, nthreads);

	
	volatile GPUGlobals* gpuGlobals;
	initializer(&gpuGlobals);

	double time_before = _timestamp();
	
	test_use<<<nblocks,nthreads,0,gpuGlobals->streamMgr->kernelStream>>>();

	run_gpufs_handler(gpuGlobals, device);
	cudaError_t error = cudaDeviceSynchronize();
	double time_after = _timestamp();
	double total_time = (time_after - time_before);
	fprintf(stderr, "total time: %lf s\n", total_time / (1e6));///9));

	//Check for errors and failed asserts in asynchronous kernel launch.
	if (error != cudaSuccess)
	{
		printf("Device failed, CUDA error message is: %s\n\n",
				cudaGetErrorString(error));
	}

	//PRINT_DEBUG;

	fprintf(stderr, "\n");
	delete gpuGlobals;

	cudaDeviceReset();

	return 0;
}


