#include "gpukv/fs_calls.cu.h"
#include "gpukv/host_loop.h"
#include "gpukv/util.cu.h"

#include "key-p2p.h"
#include "key_api.h"

// gardenia
#include "gardenia/timer.h"
#include "gardenia/worklistc.h"
#include "gardenia/common.h"

#include "bfs.cu.h"

#define TOTAL_VERTICES 1971281
//#define TOTAL_VERTICES 20

static __inline__ unsigned long long rdtsc(void){
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc": "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo | ( ((unsigned long long)hi) << 32) );
}

int main(int argc, char **argv){
	int num_thread = 1, handle = 1, call = 0;
	int device = 0;
	int gpu_thread = 1024, gpu_block = 28;
	int p2p_size = (192*1024*1024);
	int batch = 32;

	int tot_vert = TOTAL_VERTICES;
	unsigned int request_size=4096;
	int request_num, request_mode;

	while(1){
		int c;
		c=getopt(argc,argv,"s:n:m:p:i:b:t:c:q:h");
		if(c==-1) break;
		switch(c){
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
			case 'v':
				tot_vert=strtol(optarg, NULL, 0);
				break;
			case 'h':
				printf("syntax: %s -p <num_polling_thread> -i <process:one-0,batch_async-1,batch_sync-2> -q <batch_size> -b <gpu block> -t < gpu theads in block> -c <call:0-thread,1-warp,2-block> -v <# of vertex> -h\n",argv[0]);
				exit(1);
				break;
			///// Unused
			case 's':
				request_size=strtol(optarg,NULL,0);
				break;
			case 'n':
				request_num=strtol(optarg,NULL,0);
				break;
			case 'm':
				request_mode=strtol(optarg,NULL,0);
				break;

			default:
				printf("ERROR: invalid option\n");
				exit(1);
				break;
		}
	}

	//////////// GPU setting

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

	ASSERTDRV(cuMemAlloc(&gpu_memory, p2p_size));
	ASSERTDRV(cuMemsetD8(gpu_memory, 0x00, p2p_size));
	PRINT_TIMES("device ptr: 0x%lx \n",gpu_memory);
	unsigned int flag=1;
	ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, gpu_memory));
	//return 0;
	struct I_KEY_PIN_PARAM pin;
	struct I_KEY_UNPIN_PARAM unpin;
	int pin_fd;

	pin_fd=open(D,O_RDWR);
	pin.addr=gpu_memory;
	pin.size=p2p_size;
	pin.id=0;
	int ret=ioctl(pin_fd,I_KEY_PIN_BUF,&pin);
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


	////////// KV open
	int key_ret=key_open(F_NAME);

	volatile GPUGlobals* gpuGlobals;
	initializer(&gpuGlobals,num_thread);
	gpuGlobals->user_put=user_put;
	gpuGlobals->cuda_put=cuda_put;

	//int n=sqrt(gpu_thread);
	//dim3 nthreads(n,n);
	int nthreads = 16;

	int m = tot_vert;
	printf("[DEBUG:gpukv_bfs] Total vertices: %d\n", m);
	int *column_indices;
	DistT *d_dist;
	DistT *h_dist;
	cudaError_t error;

	/************ Kernel No finish version ***********/
	int src = 0;
	int nitems = 1;
	int nblocks;
	nthreads = BLOCK_SIZE;
	//int nthreads = BLOCK_SIZE;
	DistT zero = 0;
	DistT one = 1;
	//Worklist2 queue1(m), queue2(m);
	Worklist2 queues[2];
	struct g_flag h_flag = {.idx=0, .nitems=1, .remain=1, .round=0};
	struct g_flag *d_flag;
	queues[0] = Worklist2(m);
	queues[1] = Worklist2(m);
	Worklist2 *d_queues;

	/////// Init distance
	h_dist = (DistT *)malloc(sizeof(DistT)*m);
	for(int i = 0; i<m; ++i)
		h_dist[i] = MYINFINITY;


	Timer t;
	t.Start();
    unsigned long long clock_start = rdtsc();
	/////// Data transmission & launch kernel
		insert<<<1, 1>>>(src, queues[0]);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		CUDA_SAFE_CALL(cudaMalloc((void**)&d_queues, 2 * sizeof(queues[0])));
		CUDA_SAFE_CALL(cudaMalloc((void**)&column_indices, m * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_dist, m * sizeof(DistT)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_flag, sizeof(struct g_flag)));
		CUDA_SAFE_CALL(cudaMemcpy(d_flag, &h_flag, sizeof(struct g_flag), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m*sizeof(DistT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(&d_dist[src], &zero, sizeof(zero), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_queues, queues, 2*sizeof(queues[0]), cudaMemcpyHostToDevice));

		//bfs_kernel<<<gpu_block, nthreads, 0, gpuGlobals->streamMgr->kernelStream>>>(cuda_memory,request_size,m,column_indices, d_dist, d_queues, d_flag);
		inner_bfs<<<1,1,0,gpuGlobals->streamMgr->kernelStream>>>(cuda_memory, request_size, m, column_indices, d_dist, d_queues);

	///////// call gpufs handler	
	run_gpufs_handler(gpuGlobals, device,pin_fd, num_thread,handle,batch,call);
	
	error = cudaDeviceSynchronize();

    unsigned long long clock_end = rdtsc();
	t.Stop();
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	error = cudaDeviceSynchronize();
	/*for(int i = 0; i < (m > 100 ? 20 : m); i++){
		printf("dist[%d] = %u\n", i, h_dist[i]);
	}*/
	printf("Calculation time: %f ms.\n", t.Millisecs());
    printf("clock total: %llu\n", clock_end - clock_start);
	
	if (error != cudaSuccess)
	{
		printf("Device failed, CUDA error message is: %s\n\n",
				cudaGetErrorString(error));
	}

	PRINT_TIMES("\n");
	delete gpuGlobals;

	PRINT_TIMES("unpinning\n");
	ASSERT_EQ(ioctl(pin_fd,I_KEY_UNPIN_BUF,&unpin),0);
	ASSERTDRV(cuMemFree(gpu_memory));

	CUDA_SAFE_CALL(cudaFree(d_queues));
	CUDA_SAFE_CALL(cudaFree(column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	CUDA_SAFE_CALL(cudaFree(d_flag));
	free(h_dist);

	close(pin_fd);
	key_close(F_NAME);

	cudaDeviceReset();

	return 0;
}

