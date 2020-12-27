#include <stdio.h>
#include<stdlib.h>
#include <errno.h>

#include "gpukv/fs_calls.cu.h"
#include "gpukv/host_loop.h"
#include "gpukv/util.cu.h"

#include "key-p2p.h"
#include "key_api.h"

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(err)  __checkCudaErrors (err, __FILE__, __LINE__)
#endif

#include<algorithm>
#include "kernel/gpukv.cu"

#include "gardenia/timer.h"
//using namespace std;

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int) err, cudaGetErrorString(err));
		exit(-1);
	}
}

#define START_KEY 10000

int main(int argc, char** argv)
{
	int device = 0;
	int request_num=1024, request_mode=0, num_thread=1, handle=1,call=1;
	unsigned int request_size=4096;
	int gpu_thread=1024,gpu_block=28;
	int gpu_memory_size=(192*1024*1024); ///p2p mem :220MB
	int batch=32;

	int ret=0;

	printf("[DEBUG] cp.cu start\n");
	while(1){
		int c;
		c=getopt(argc,argv,"s:n:m:p:i:b:t:c:q:h");
		if(c==-1) break;
		switch(c){
			case 's':
				request_size=strtol(optarg,NULL,0);
				break;
			case 'n':
				request_num=strtol(optarg,NULL,0);
				break;
			case 'm':
				request_mode=strtol(optarg,NULL,0);
				break;
			case 'p':
				num_thread=strtol(optarg,NULL,0);
				break;
			case 'i':
				handle=strtol(optarg,NULL,0);
				break;
			case 'b':
				gpu_block=strtol(optarg,NULL,0);
				break;
			case 't':
				gpu_thread=strtol(optarg,NULL,0);
				break;
			case 'c':
				call=strtol(optarg,NULL,0);
				break;
			case 'q':
				batch=strtol(optarg,NULL,0);
				break;
			case 'h':
				printf("syntax: %s -s <request_size> -n <num_request> -m <mode:get-0,put-1,get_and_put-2,program-3> -p <num_polling_thread> -i <process:one-0,batch_async-1,batch_sync-2> -q <batch_size> -b <gpu block> -t < gpu theads in block> -c <call:0-thread,1-warp,2-block> -h\n",argv[0]);
				exit(1);
				break;
			default:
				printf("ERROR: invalid option\n");
				exit(1);
				break;
		}
	}
	printf("GPUKV | size : %d KB, num : %d, mode : %d, user_thread : %d, process_mode : %d, batch : %d, block : %d, thread : %d, gpu_mode : %d\n",
			request_size/1024, request_num, request_mode, num_thread, handle, batch, gpu_block,gpu_thread,call);
	/////////////gpu setting
	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	PRINT_TIMES("GPU id: %d, name: %s\nPCI domain: %d, bus: %d, device: %d\n",device,deviceProp.name,deviceProp.pciDomainID,deviceProp.pciBusID,deviceProp.pciDeviceID);

	PRINT_TIMES("Running on device %d: \"%s\"\n", device, deviceProp.name);

	PRINT_TIMES("\tblocks %d threads %d\n", gpu_block, gpu_thread);
	////////////////////////////////////////////////
	////////////////////p2p memory mapping, and general memory
	void *dummy;
	ASSERTRT(cudaMalloc(&dummy,0));

	CUdeviceptr gpu_memory;
	char* cuda_memory;
	char *cuda_put,*user_put;

	ASSERTDRV(cuMemAlloc(&gpu_memory,gpu_memory_size));
	ASSERTDRV(cuMemsetD8(gpu_memory, 0x00, gpu_memory_size));
	PRINT_TIMES("device ptr: 0x%lx \n",gpu_memory);
	unsigned int flag=1;
	ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, gpu_memory));
	//return 0;
	struct I_KEY_PIN_PARAM pin;
	struct I_KEY_UNPIN_PARAM unpin;
	int pin_fd;

	pin_fd=open(D,O_RDWR);
	pin.addr=gpu_memory;
	pin.size=gpu_memory_size;
	pin.id=0;
	ret=ioctl(pin_fd,I_KEY_PIN_BUF,&pin);
	if(ret!=0){printf("gpu pin error!\n"); return 0;}
	PRINT_TIMES("info.va: %lx\n",pin.start_addr);   
	PRINT_TIMES("info.mapped_size: %lu B,(%lu MB)\n",pin.pin_size,pin.pin_size/(1024*1024));
	//      OUT << "info.page_size: " << info.page_size << endl;

	// remember that mappings start on a 64KB boundary, so let's
	// calculate the offset from the head of the mapping to the
	// beginning of the buffer
	int start_off = pin.start_addr-gpu_memory;//info.va - d_A;
	PRINT_TIMES("page offset: %lx\n", start_off);
	cuda_memory=(char*)gpu_memory;


	///FOR PUT
	if(request_mode!=0){
		CUDA_SAFE_CALL(cudaHostAlloc(&user_put,gpu_memory_size,cudaHostAllocDefault));
		ASSERTRT(cudaMalloc((void**)&cuda_put,gpu_memory_size));
	}
	///////////////////////////////////////////////////////
	///////key_value open
	int key_ret=key_open(F_NAME);

	//////////////////////////////////////////////////////
	//////request list make
	int *request_list,*cuda_request;
	int i,j;
	request_list=(int*)malloc(sizeof(int)*request_num);
	for(i=0;i<request_num;i++){
		request_list[i]=i+START_KEY;
	}
	random_shuffle(request_list,request_list+request_num);
	ASSERTRT(cudaMalloc((void**)&cuda_request,sizeof(int)*request_num));
	ASSERTRT(cudaMemcpy(cuda_request,request_list,sizeof(int)*request_num,cudaMemcpyHostToDevice));
	/////////////////////////////////////////////////////
	///////push data to key_value_SSD or GPU_MEMORY
	int * cuda_reset;
	ASSERTRT(cudaMalloc((void**)&cuda_reset,sizeof(int)*gpu_block));
	ASSERTRT(cudaMemset((void*)cuda_reset,0,sizeof(int)*gpu_block));
	/*	
		char *temp_data=(char*)malloc(sizeof(char)*gpu_memory_size);
		int temp_cnt=0;
		memset(temp_data,0,gpu_memory_size);
		if(request_mode==1){//put
		for(i=0;i<gpu_memory_size;i+=request_size){
		for(j=i; j<i+request_size && j<gpu_memory_size;j++){
		temp_data[j]=temp_cnt+'0';
		}
		temp_cnt=(temp_cnt+1)%10;
		}
		ASSERTRT(cudaMemcpy(cuda_put,temp_data,gpu_memory_size,cudaMemcpyHostToDevice));
		}*/
	//////////////////////////////////////////////////////////////////////
	int limit_num;
	unsigned long real_size,real_num;
	if(call==0 || call==5) real_num=gpu_block*gpu_thread; //thread
	else if(call==1) real_num=gpu_block*gpu_thread/32; //warp
	else real_num=gpu_block; //block
	real_size=(unsigned long)real_num*request_size;
	if(real_size>gpu_memory_size){
		limit_num=gpu_memory_size/request_size;
		PRINT_TIMES("LIMIT_SIZE! %d-> Limit_num : %d\n",real_num,limit_num);
	}
	else limit_num=real_num;
	printf("GPUKV | Real Used GPUMEM size : %lf MB, real_num : %d, limit_num : %d\n",(double)real_size/(1024*1024),real_num,limit_num);
	///////////////////////////////////////////////////////////////////////////////////////////
	////////ready for start


	volatile GPUGlobals* gpuGlobals;
	initializer(&gpuGlobals,num_thread);
	gpuGlobals->user_put=user_put;
	gpuGlobals->cuda_put=cuda_put;

	int n=sqrt(gpu_thread);
	dim3 nthreads(n,n);

	double time_before = _timestamp();

	if(request_mode==0)
		test_get<<<gpu_block,nthreads,0,gpuGlobals->streamMgr->kernelStream>>>(cuda_memory,cuda_put,cuda_request,request_num,request_size,call,gpu_memory_size,limit_num);
	else if(request_mode==1)
		test_put<<<gpu_block,nthreads,0,gpuGlobals->streamMgr->kernelStream>>>(cuda_memory,cuda_put,cuda_request,request_num,request_size,call,gpu_memory_size,limit_num);
	else if(request_mode==2)
		test_modify<<<gpu_block,nthreads,0,gpuGlobals->streamMgr->kernelStream>>>(cuda_memory,cuda_put,cuda_request,request_num,request_size,gpu_memory_size,limit_num);
	else if(request_mode==3)
		test_app<<<gpu_block,nthreads,0,gpuGlobals->streamMgr->kernelStream>>>(cuda_memory,cuda_put,cuda_request,request_num,request_size,gpu_memory_size,limit_num);
	else{
		test_modify2<<<gpu_block,nthreads,0,gpuGlobals->streamMgr->kernelStream>>>(cuda_memory,cuda_put,cuda_request,cuda_reset, request_num,request_size,gpu_memory_size,limit_num);
	}
	printf("[DEBUG] run gpufs handler call\n");
	run_gpufs_handler(gpuGlobals, device,pin_fd, num_thread,handle,batch,call);

	cudaError_t error = cudaDeviceSynchronize();
	double time_after = _timestamp();
	double total_time = elapsed_time(time_before,time_after);
	printf("GPUKV | End total time: %lf s\n\n\n", total_time);///9));
	//Check for errors and failed asserts in asynchronous kernel launch.
	if (error != cudaSuccess)
	{
		printf("Device failed, CUDA error message is: %s\n\n",
				cudaGetErrorString(error));
	}

	///////////////////////////////////////////
	/////All Freeeeee
	PRINT_TIMES("\n");
	delete gpuGlobals;

	PRINT_TIMES("unpinning\n");
	ASSERT_EQ(ioctl(pin_fd,I_KEY_UNPIN_BUF,&unpin),0);
	ASSERTDRV(cuMemFree(gpu_memory));

	CUDA_SAFE_CALL(cudaFree(cuda_request));
	if(request_mode!=0){
		CUDA_SAFE_CALL(cudaFreeHost(user_put));
		CUDA_SAFE_CALL(cudaFree(cuda_put));
	}
	/*
	if(bfs){
		CUDA_SAFE_CALL(cudaFree(column_indices));
		CUDA_SAFE_CALL(cudaFree(d_dist));
	}*/

	CUDA_SAFE_CALL(cudaFree(cuda_reset));
	free(request_list);

	//	free(temp_data);

	close(pin_fd);
	key_close(F_NAME);

	cudaDeviceReset();

	return 0;
}
