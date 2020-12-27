#include <stdio.h>
#include <stdlib.h>

#include <random>

#include "gpukv/fs_calls.cu.h"
#include "gpukv/host_loop.h"
#include "gpukv/util.cu.h"

#include "key-p2p.h"
#include "key_api.h"

#include "kernel.cu.h"

#include <curand.h>

const char *(mode_name[]) = {"streaming","dynamic"};

static __inline__ unsigned long long rdtsc(void){
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc": "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo | ( ((unsigned long long)hi) << 32) );
}

int main(int argc, char **argv){
	int device = 0;
	int request_num=32*1024*128; // 32K * 128
    int workload=0, num_thread=1, handle=1,call=1;
	unsigned int request_size=4096; // 4K
	int gpu_thread=256,gpu_block=256;
	int gpu_memory_size=(4*1024*1024); ///p2p mem :220MB
	int batch=32;

	printf("[DEBUG] Synthetic start\n");
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
				workload=strtol(optarg,NULL,0);
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
    printf("[GPUKV        ] vertex: %d, mode: %s, batch: %d\n", request_num, mode_name[workload/2-1], batch);
	//////////// GPU setting

	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));

	////////////////////////////////////////////////
	////////////////////p2p memory mapping
	void *dummy;
	ASSERTRT(cudaMalloc(&dummy,0));

	CUdeviceptr gpu_memory;
	char* cuda_memory;

	ASSERTDRV(cuMemAlloc(&gpu_memory, gpu_memory_size));
	ASSERTDRV(cuMemsetD8(gpu_memory, 0x00, gpu_memory_size));
	PRINT_TIMES("device ptr: 0x%lx \n",gpu_memory);
	unsigned int flag=1;
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

	cuda_memory=(char*)gpu_memory;


	////////// KV open
	int key_ret=key_open(F_NAME);

    //////////////////// GPU setup
    //int n = sqrt(gpu_thread);
    //dim3 nthreads(n, n);
	char *cuda_put,*user_put;
    CUDA_SAFE_CALL(cudaHostAlloc(&user_put,gpu_memory_size,cudaHostAllocDefault));
    ASSERTRT(cudaMalloc((void**)&cuda_put,gpu_memory_size));

    // gpufs handler init
	volatile GPUGlobals* gpuGlobals;
	initializer(&gpuGlobals,num_thread);
	gpuGlobals->user_put=user_put;
	gpuGlobals->cuda_put=cuda_put;

	//int nthreads = sqrt(gpu_thread);

	cudaError_t error;

    /////////////////////////////////////// GPU kernel ///////////
    int stride = gpu_memory_size / request_size;
    gpu_block = stride / gpu_thread;

    /****************** request list make ********************/
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<unsigned long long> distribution(0, request_num-1);

	int *request_list,*cuda_request;
	int i;
	request_list=(int*)malloc(sizeof(int)*stride);
	for(i=0;i<stride;i++){
		request_list[i]= distribution(generator);
	}
    ASSERTRT(cudaMalloc((void**)&cuda_request,sizeof(int)*stride));
	ASSERTRT(cudaMemcpy(cuda_request,request_list,sizeof(int)*stride,cudaMemcpyHostToDevice));

    /********* random ***********/
    curandState_t* states;
    cudaMalloc((void**)&states, sizeof(curandState_t)*stride);
    init_curand_state<<<gpu_block, gpu_thread>>>(states);

    /********* gpu kernel start *********/
    unsigned long long clock_start, clock_end;
    double time_before = _timestamp();
    clock_start = rdtsc();
    int iter = 3;
    double portion = 1;
    //int *alive, *alive2;
    //int host_alive, host_alive2;

    //cudaMalloc((void**)&alive, sizeof(int));
    //cudaMalloc((void**)&alive2, sizeof(int));

    switch(workload){
        case 2:
            {
                //for(int i = 0; i < request_num / stride; ++i){ 
                //for(int i = 0; i < 2; ++i){ 
            streaming_gpukv<<<gpu_block, gpu_thread, 0, gpuGlobals->streamMgr->kernelStream>>>(
                //cuda_memory, request_size, request_num, stride, iter, portion, alive, alive2);
                cuda_memory, request_size, request_num, stride, iter, portion);
            //streaming_gpukv<<<gpu_block, gpu_thread, 0, gpuGlobals->streamMgr->kernelStream>>>(
                //cuda_memory, request_size, request_num, i*stride, iter, portion, alive, alive2);

        	//run_gpufs_handler(gpuGlobals, device,pin_fd, num_thread,handle,batch,call);
            //error = cudaDeviceSynchronize();
                //}
            }
            break;
        case 4:
            portion = 0.90625;
            dynamic_gpukv<<<1, 1, 0, gpuGlobals->streamMgr->kernelStream>>>(
                cuda_memory, request_size, cuda_request, request_num, stride, 
                iter, portion, gpu_block, gpu_thread, states);
             break;
    }


	run_gpufs_handler(gpuGlobals, device,pin_fd, num_thread,handle,batch,call);

    error = cudaDeviceSynchronize();
    if(error){
    }
    double time_after = _timestamp();
    clock_end = rdtsc();

    double total_time = elapsed_time(time_before, time_after);
    printf("GPUKV iter: %d, portion: %.2f, stride: %d | End total time: %lf s\n\n", iter, portion, stride, total_time);
    printf("clock total: %llu\n", clock_end - clock_start);
    //cudaMemcpy(&host_alive, alive, sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(&host_alive2, alive2, sizeof(int), cudaMemcpyDeviceToHost);
    //printf("%d %d\n", host_alive, host_alive2);



    //////////////////////////////////////////////////////////////


    ////////////////////////////
    ///////// Finishing
    delete gpuGlobals;

    //unpinning
	ASSERT_EQ(ioctl(pin_fd,I_KEY_UNPIN_BUF,&unpin),0);
    ASSERTDRV(cuMemFree(gpu_memory));

    // memory free
	CUDA_SAFE_CALL(cudaFree(cuda_request));
    CUDA_SAFE_CALL(cudaFreeHost(user_put));
    CUDA_SAFE_CALL(cudaFree(cuda_put));

	free(request_list);

    // kv close
    close(pin_fd);

	key_close(F_NAME);

	cudaDeviceReset();
    return 0;
}
