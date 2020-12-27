#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "key-p2p.h"
#include "key_api.h"
#include "gpukv/timer.h"

#include <algorithm>

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



int main(int argc, char** argv){
	srand(time(NULL));
	int device = 0;
	int gpu_memory_size = 192*1024*1024;


	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	
	void *dummy;
	ASSERTRT(cudaMalloc(&dummy,0));

	CUdeviceptr gpu_memory;
	ASSERTDRV(cuMemAlloc(&gpu_memory,gpu_memory_size));
	ASSERTDRV(cuMemsetD8(gpu_memory, 0x00, gpu_memory_size));

	unsigned int flag = 1;
	ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, gpu_memory));

	struct I_KEY_PIN_PARAM pin;
	struct I_KEY_UNPIN_PARAM unpin;
	int pin_fd;

	pin_fd=open(D,O_RDWR);
	pin.addr=gpu_memory;
	pin.size=gpu_memory_size;
	pin.id=0;
	int ret=ioctl(pin_fd,I_KEY_PIN_BUF,&pin);
	if(ret!=0){printf("gpu pin error!\n"); return 0;}

	
	int key_ret=key_open(F_NAME);

	int *request_list;
	int start_key = 0;
	int request_num = 10;
	int i;
	request_list=(int*)malloc(sizeof(int)*request_num);
	for(i=0;i<request_num;i++){
		request_list[i]=i+start_key;
		request_list[i] += 10000*i;
	}
	random_shuffle(request_list,request_list+request_num);
	printf("aaaa\n");
	
	// KEY_PAGE_SIZE: 4096
	int buffer_size=KEY_PAGE_SIZE*1024;
	int request_size = 1024*512;
	char *buffer[10];
	for(i = 0; i < request_num; i++){
		char num = rand() % 10 + '0';
		if(posix_memalign((void**)&buffer[i],KEY_PAGE_SIZE,buffer_size)){
			printf("can not allocate io payload buffer!\n");
			return 0;
		}
		memset(buffer[i], num, buffer_size);
	}
	
	printf("setup data at ssd start\n");

	// setup data at ssd for pinned memory
	char tmp[524288] = {0};
	for(i = 0; i < request_num; ++i){
		// mode, key, buf, length
		ret = key_op(KEY_PUT, request_list[i], buffer[request_list[i] % 10], request_size);
		printf("key - %d PUT:\n", request_list[i]);
		if(ret != request_size) printf("error! ret=%d\n", ret);
		for(int j = 0; j < 10; ++j)
			printf("%c", buffer[request_list[i] % 10][j]);
		printf("\n");
		/*printf("GET:\n");
		ret = key_op(KEY_GET, request_list[i], tmp, request_size);
		for(int j = 0; j < 10; ++j)
			printf("%c", buffer[i][j]);
		printf("\n\n");*/
	}
	printf("put complete\n");
	getchar();

	// load data at pinned memory
	/*for(i = 0; i < request_num; ++i) memset(buffer[i], 0, buffer_size);
	printf("aaa\n");
	for(i = 0; i < request_num; ++i){
		ret = key_p2p_op(KEY_GET, request_list[i], i*request_size, request_size);
		//request_list[i] += start_key;
		printf("%d\n", i);
	}
	printf("p2p ready complete\n");*/
	/*cuMemcpyDtoH(buffer, gpu_memory, request_size*request_num);
	printf("p2p check\n");
	for(i = 0; i < request_num; ++i){
		for(int j = 0; j < 10; ++j){
			printf("%d ", buffer[i*request_size/buffer_size][i*request_size % buffer_size + j]);
		}
		printf("\n");
	}*/


	printf("gpu pinned memory data setup complete\n");
	getchar();

	// p2p put test
	//for(i = 0; i < request_num; ++i){
//		ret = key_p2p_op(KEY_PUT, request_list[i], i*request_size, request_size);
//		printf("put p2p ret: %d\n", ret);
//	}

	for(i = 0; i < request_num; ++i)
	{
		memset(buffer[i], 0, buffer_size);
		for(int j = 0; j< 10; ++j)
			printf("%02X", (unsigned char)buffer[i][j]);
		printf("\n");
	}
	for(i = 0; i < request_num; ++i){
		ret = key_op(KEY_GET, request_list[i], buffer[i], request_size);
		printf("key - %d get ret: %d\n", request_list[i], ret);
		for(int j = 0; j < 10; ++j)
			printf("%c", buffer[i][j]);
		printf("\n");
	}


	printf("good\n");
	//key_close(F_NAME);
	//close(pin_fd);

	/*ASSERT_EQ(ioctl(pin_fd,I_KEY_UNPIN_BUF,&unpin),0);
	ASSERTDRV(cuMemFree(gpu_memory));
	
	key_close(F_NAME);
	close(pin_fd);
	
	cudaDeviceReset();
	
	free(request_list);
	for(i = 0; i < request_num; ++i){
		free(buffer[i]);
	}*/

	return 0;
}
