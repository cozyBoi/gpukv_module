#include <stdio.h>
#include<stdlib.h>
#include <errno.h>

#include "gpukv/util.cu.h"

#include "key-p2p.h"
#include "key_api.h"

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(err)  __checkCudaErrors (err, __FILE__, __LINE__)
#endif

#include<algorithm>
#include<pthread.h>

#include"kernel/conventional.cu"

//1using namespace std;

#ifdef TRACE

#define PRINT_TRACE(...) fprintf(stderr, __VA_ARGS__);

#else

#define PRINT_TRACE(...)

#endif

#ifdef TIMING_STATS

#define PRINT_TIMES(...) fprintf(stderr, __VA_ARGS__);

#else

#define PRINT_TIMES(...)

#endif


inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int) err, cudaGetErrorString(err));
		exit(-1);
	}
}

#define START_KEY 10000
#define START_PUT 1000000
#define MAX_CHUNK_NUM 8 //64
#define IO_NUM 4

pthread_t rwLoopTasksIDs[3][MAX_CHUNK_NUM];

volatile int done = 0;
//pthread_mutex_t rwLoopTasksLocks[RW_HOST_WORKERS];
//pthread_cond_t  rwLoopTasksConds[RW_HOST_WORKERS];
//pthread_t rwLoopTasksIDs[RW_HOST_WORKERS];
//TaskData rwLoopTasksData[RW_HOST_WORKERS];
typedef struct _pipe{
	int id;//0:read , 1: write, 2: chunk
	int io_num; //

	int num_thread;
	int chunk_cnt; //chunk 당 request 개수
	int request_num; //전체 request 개수
	unsigned int request_size; //
	int *request_list; //request list
	char *ptr;
	int call;
}Pipe;

Pipe task_data[2][MAX_CHUNK_NUM];

typedef struct _chunk{
	int id;

	int chunk_num;
	int chunk_cnt;
	int request_num;
	unsigned int request_size,out_request_size;
	int *request_list;
	int request_mode;
	char *user_get,*user_put,*get_pin,*put_pin;

	int cuda_block;
	int gpu_thread;

}Chunk;

Chunk chunk_data[MAX_CHUNK_NUM];

unsigned long transfer_size[2][MAX_CHUNK_NUM]={0};



pthread_mutex_t cond_lock[2][MAX_CHUNK_NUM]; //for chunk process
pthread_cond_t cond[2][MAX_CHUNK_NUM]; //for chunk process
volatile int request_status[2][MAX_CHUNK_NUM]={0};
/////////////////////////////////////////////
volatile int io_complete[2]={0};
pthread_mutex_t io_lock[2]; //for io
pthread_cond_t io_cond[2]; //for io


cudaStream_t* stream;
cudaEvent_t* event[3];
int *cuda_reset;
char* cuda_memory;
char *cuda_put;

void* pipe_task(void* param){ ///only IO thread
	Pipe* s=(Pipe*)param;

	int id=s->id;
	int io_num=s->io_num;

	int chunk_cnt=s->chunk_cnt;
	int request_num=s->request_num;
	int* request_list=s->request_list;
	unsigned int request_size=s->request_size;
	int call=s->call;
	char *buffer=s->ptr;
	unsigned long chunk_size=(unsigned long)request_size*chunk_cnt;
	int i,j;
	int ret;

	int num_chunk=s->num_thread;

	int io_cnt=chunk_cnt/IO_NUM;
	int current_chunk=0;
	int chunk_request_start=0;
	int current_request_start=chunk_request_start+io_cnt*io_num;

	unsigned long start;
	
	char *temp_buffer;
	if(posix_memalign((void**)&temp_buffer,KEY_PAGE_SIZE,request_size)){
		printf("can not allocate io payload buffer!\n");
		return NULL;
	}

//		printf("thread %d,%d start!\n",id,io_num);
//		return NULL;
	while(!done){
		if(current_request_start>=request_num){
			pthread_mutex_lock(&io_lock[id]);
			io_complete[id]++;
			pthread_mutex_unlock(&io_lock[id]);
			break;
		}

		//먼저 돌아도 되는지 체크
		pthread_mutex_lock(&cond_lock[id][current_chunk]);
		//0:not yet, 1 : go process, 2: complete
		if(request_status[id][current_chunk]!=1) pthread_cond_wait(&cond[id][current_chunk],&cond_lock[id][current_chunk]);
		pthread_mutex_unlock(&cond_lock[id][current_chunk]);

		int end=min(current_request_start+io_cnt,request_num);

		if(id==0){//GET
			start=current_chunk*chunk_size+(io_cnt*io_num)*(unsigned long)request_size;
			for(j=current_request_start;j<end;j++){
				if(call==0){//no p2p
					ret=key_op(KEY_GET,request_list[j],temp_buffer,request_size);
					memcpy(buffer+start,temp_buffer,request_size);
					//					ret=key_op(KEY_GET,request_list[j],buffer+(i*chunk_size)+(start*request_size),request_size);
				}
				else{
					ret=key_p2p_op(KEY_GET,request_list[j],(i*chunk_size)+(start*request_size),request_size);
				}
				if(ret<=0) printf("key_op error!\n");
				start+=request_size;
			}
			transfer_size[id][io_num]+=(unsigned long)(end-current_request_start)*request_size;

		}
		else{///PUT
			start=current_chunk*chunk_size+(io_cnt*io_num)*(unsigned long)request_size;
			for(j=current_request_start;j<end;j++){
				memcpy(temp_buffer,buffer+start,request_size);
				ret=key_op(KEY_PUT, request_list[j]+START_PUT ,temp_buffer,request_size);
				//				ret=key_op(KEY_GET,request_list[j],buffer+(i*chunk_size)+(start*request_size),request_size);

				if(ret<=0) printf("key_op error!\n");
				start+=request_size;
			}
			transfer_size[id][io_num]+=(unsigned long)(end-current_request_start)*request_size;
		}
		//////////chunk complete
		pthread_mutex_lock(&io_lock[id]);
		io_complete[id]++;
		if(io_complete[id]<IO_NUM){
			pthread_cond_wait(&io_cond[id],&io_lock[id]);
		}
		else{
			io_complete[id]=0;
			pthread_cond_broadcast(&io_cond[id]);

			pthread_mutex_lock(&cond_lock[id][current_chunk]); //wake up
			//0:not yet, 1 : go process, 2: complete
			request_status[id][current_chunk]=2;
			pthread_cond_broadcast(&cond[id][current_chunk]);
			pthread_mutex_unlock(&cond_lock[id][current_chunk]);
		}
		pthread_mutex_unlock(&io_lock[id]);

		current_chunk=(current_chunk+1)%num_chunk;
		chunk_request_start+=chunk_cnt;
		current_request_start=chunk_request_start+io_cnt*io_num;

	}

	free(temp_buffer);
	return NULL;
}
void* chunk_task(void* param){ ///only IO thread
	Chunk* s=(Chunk*)param;

	int current_chunk=s->id;

	int chunk_cnt=s->chunk_cnt;
	int request_num=s->request_num;
	int* request_list=s->request_list;
	unsigned int request_size=s->request_size;
	unsigned int out_request_size=s->out_request_size;
	char *user_get=s->user_get;
	char *user_put=s->user_put;
	char *get_pin=s->get_pin;
	char *put_pin=s->put_pin;
	unsigned long chunk_size=(unsigned long)request_size*chunk_cnt;
	unsigned long out_size=(unsigned long)out_request_size*chunk_cnt;
	int i,j;
	int ret;

	int chunk_num=s->chunk_num;
	int request_mode=s->request_mode;

	int request_start=chunk_cnt*current_chunk;
	int next_key=chunk_cnt*chunk_num;

	unsigned long start,out_start;
	unsigned long transfer_size,out_transfer_size;

	int cuda_block=s->cuda_block;
	int gpu_thread=s->gpu_thread;
	int n=sqrt(gpu_thread);
	dim3 nthreads(n,n);
	int cuda_cnt=1;

	while(!done){
//		printf("chunk %d | start_request : %d, chunk_cnt: %d\n",current_chunk,request_start,chunk_cnt);
//		return NULL;
		if(request_start>=request_num) break;
		///////////GET
//		printf("GET\n");
		pthread_mutex_lock(&cond_lock[0][current_chunk]);
		if(request_status[0][current_chunk]!=2) pthread_cond_wait(&cond[0][current_chunk],&cond_lock[0][current_chunk]);
		request_status[0][current_chunk]=0; //init
		pthread_mutex_unlock(&cond_lock[0][current_chunk]);

		int remain=min(chunk_cnt,request_num-request_start);
		transfer_size=(unsigned long)request_size*remain;

		start=chunk_size*current_chunk;
		out_start=out_size*current_chunk;
		out_transfer_size=(unsigned long)out_request_size*remain;


//		memcpy(get_pin+start,user_get+start,transfer_size);
		CUDA_SAFE_CALL(cudaMemcpy(cuda_memory+start,user_get+start,transfer_size,cudaMemcpyHostToDevice));
		////here
//		printf("KERNEL\n");
		if(request_mode==0){
			request_start+=next_key;
			pthread_mutex_lock(&cond_lock[0][current_chunk]);
			request_status[0][current_chunk]=1;
			pthread_cond_broadcast(&cond[0][current_chunk]);
			pthread_mutex_unlock(&cond_lock[0][current_chunk]);
		}
		else{//2,3,4
			/////////////KERNEL
			if(request_mode==2){
				test_modify<<<cuda_block,nthreads>>>(cuda_memory+start,cuda_put+out_start,remain,request_size);
			}
			else if(request_mode==3){
				test_app<<<cuda_block,nthreads>>>(cuda_memory+start,cuda_put+out_start,remain,request_size);
			}
			else if(request_mode==4){
				test_modify2<<<cuda_block,nthreads>>>(cuda_memory+start,cuda_put+out_start,cuda_reset,cuda_cnt, remain,request_size);
				cuda_cnt++;
			}

			////////////////////////////////////////////////////////////
			///////////PUT
			CUDA_SAFE_CALL(cudaMemcpy(user_put+out_start,cuda_put+out_start,out_transfer_size,cudaMemcpyDeviceToHost));

//			printf("PUT\n");
			///////////////////////////////////////////////////////////
			pthread_mutex_lock(&cond_lock[1][current_chunk]);
			//			if(request_status[1][current_chunk]==1) pthread_cond_wait(&cond[1][current_chunk],&cond_lock[1][current_chunk]);
//			memcpy(user_put+out_start,put_pin+out_start,out_transfer_size);
			request_status[1][current_chunk]=1;
			pthread_cond_broadcast(&cond[1][current_chunk]);
			pthread_mutex_unlock(&cond_lock[1][current_chunk]);

			//////////WAIT
			pthread_mutex_lock(&cond_lock[1][current_chunk]);
			if(request_status[1][current_chunk]!=2) pthread_cond_wait(&cond[1][current_chunk],&cond_lock[1][current_chunk]);
			request_status[1][current_chunk]=0;
			pthread_mutex_unlock(&cond_lock[1][current_chunk]);

//			printf("COMPLETE\n");
			request_start+=next_key;

			///GET_READY
			pthread_mutex_lock(&cond_lock[0][current_chunk]);
			request_status[0][current_chunk]=1;
			pthread_cond_broadcast(&cond[0][current_chunk]);
			pthread_mutex_unlock(&cond_lock[0][current_chunk]);
		}
	}
	return NULL;
}

int main(int argc, char** argv)
{
	int device = 0;
	int request_num=1024, request_mode=2, num_thread=1, handle=1,call=0;
	unsigned int request_size=1024*1024;
	int gpu_thread=64,gpu_block=256;
	int gpu_memory_size=(192*1024*1024); ///p2p mem :220MB
	int batch=32;

	int ret=0;
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
	//	printf("1111111111111\n");
	void *dummy;
	ASSERTRT(cudaMalloc(&dummy,0));

	CUdeviceptr gpu_memory;
	char *user_put,*user_get,*get_pin=NULL,*put_pin=NULL;

	unsigned int flag=1;
	struct I_KEY_PIN_PARAM pin;
	struct I_KEY_UNPIN_PARAM unpin;
	int pin_fd;
	int start_off;

	//	printf("1111111111111\n");

	if(call==1){//for P2P
		ASSERTDRV(cuMemAlloc(&gpu_memory,gpu_memory_size));
		ASSERTDRV(cuMemsetD8(gpu_memory, 0x00, gpu_memory_size));
		PRINT_TIMES("device ptr: 0x%lx \n",gpu_memory);
		ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, gpu_memory));

		pin_fd=open(D,O_RDWR);
		pin.addr=gpu_memory;
		pin.size=gpu_memory_size;
		pin.id=0;
		ret=ioctl(pin_fd,I_KEY_PIN_BUF,&pin);
		if(ret!=0){printf("gpu pin error!\n"); return 0;}
		PRINT_TIMES("info.va: %lx\n",pin.start_addr);   
		PRINT_TIMES("info.mapped_size: %lu B,(%lu MB)\n",pin.pin_size,pin.pin_size/(1024*1024));

		start_off = pin.start_addr-gpu_memory;//info.va - d_A;
		PRINT_TIMES("page offset: %lx\n", start_off);
		cuda_memory=(char*)gpu_memory;
	}
	else{
		user_get=(char*)malloc(sizeof(char)*gpu_memory_size);
		//		CUDA_SAFE_CALL(cudaHostAlloc(&get_pin,gpu_memory_size,cudaHostAllocDefault));
		ASSERTRT(cudaMalloc((void**)&cuda_memory,gpu_memory_size)); //gpu input memory
	}
	///FOR PUT
	if(request_mode!=0){
		user_put=(char*)malloc(sizeof(char)*gpu_memory_size);
		//		CUDA_SAFE_CALL(cudaHostAlloc(&put_pin,gpu_memory_size,cudaHostAllocDefault));
		ASSERTRT(cudaMalloc((void**)&cuda_put,gpu_memory_size));
	}



	///////////////////////////////////////////////////////
	///////key_value open
	int key_ret=key_open(F_NAME);

	//////////////////////////////////////////////////////
	//////request list make
	int *request_list;
	int i,j;
	request_list=(int*)malloc(sizeof(int)*request_num);
	for(i=0;i<request_num;i++){
		request_list[i]=i+START_KEY;
	}
	random_shuffle(request_list,request_list+request_num);

	ASSERTRT(cudaMalloc((void**)&cuda_reset,sizeof(int)*gpu_block));
	ASSERTRT(cudaMemset((void*)cuda_reset,0,sizeof(int)*gpu_block));
	//	}
	/////////////////////////////////////////////////////
	///////push data to key_value_SSD or GPU_MEMORY

	////////////////////////START
	int chunk_num=num_thread;
	int n=sqrt(gpu_thread);
	dim3 nthreads(n,n);
	int cuda_block=gpu_block/num_thread;
	double time_before,time_after;

	////////ready for START_KEY
	int chunk_cnt=(gpu_memory_size/num_thread)/request_size;
	int next_key=chunk_cnt*num_thread;
	unsigned long chunk_size=(unsigned long)chunk_cnt*request_size;
	unsigned long out_size=0;
	unsigned int out_request_size=0;


	if(request_mode==3){ out_size=chunk_size/3; out_request_size=request_size/3;} //image
	else{ out_size=chunk_size; out_request_size=request_size;}

	printf("SIMPLE | size : %d KB, num : %d, mode : %d, user_thread : %d, block : %d, thread : %d, p2p_mode : %d\n",	request_size/1024, request_num,request_mode, num_thread, gpu_block,gpu_thread,call);
	printf("SIMPLE | user_thread : %d, chunk_size : %lf MB, chunk_request_cnt : %d, next_key %d\n",num_thread, (double)chunk_size/(1024*1024),chunk_cnt,next_key);

	for(i=0;i<chunk_num;i++){

		pthread_cond_init(&(cond[0][i]),0);
		pthread_cond_init(&(cond[1][i]),0);
		pthread_mutex_init(&(cond_lock[0][i]),0);
		pthread_mutex_init(&(cond_lock[1][i]),0);
	}




	for(i=0;i<2;i++){
		pthread_mutex_init(&(io_lock[i]),0);
		pthread_cond_init(&io_cond[i],0);
	}

	pthread_attr_t attr;
	pthread_attr_init( &attr );
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );

	////io
	for(i=0;i<2;i++){
		for(j=0;j<IO_NUM;j++){
			task_data[i][j].id=i; //for DISK read threads
			task_data[i][j].io_num=j;
			task_data[i][j].num_thread=num_thread;
			task_data[i][j].chunk_cnt=chunk_cnt;
			task_data[i][j].request_num=request_num;
			if(i==0){
				task_data[i][j].request_size=request_size;
				if(call==0) task_data[i][j].ptr=user_get;
				else task_data[i][j].ptr=NULL;
			}
			else if(i==1){
				task_data[i][j].request_size=out_request_size;
				task_data[i][j].ptr=user_put;
			}
			task_data[i][j].request_list=request_list;
			task_data[i][j].call=call;
		}
		io_complete[i]=0;
	}
	//////////chunk
	for(i=0;i<chunk_num;i++){
		chunk_data[i].id=i;
		chunk_data[i].chunk_num=chunk_num;
		chunk_data[i].chunk_cnt=chunk_cnt;
		chunk_data[i].request_num=request_num;
		chunk_data[i].request_size=request_size;
		chunk_data[i].out_request_size=out_request_size;
		chunk_data[i].request_list=request_list;
		chunk_data[i].user_get=user_get;
		chunk_data[i].get_pin=get_pin;
		chunk_data[i].user_put=user_put;
		chunk_data[i].put_pin=put_pin;
		chunk_data[i].request_mode=request_mode;

		chunk_data[i].cuda_block=cuda_block;
		chunk_data[i].gpu_thread=gpu_thread;

		request_status[0][i]=1;
		request_status[1][i]=0;
	}

	time_before= _timestamp();

	for(i=0;i<IO_NUM;i++){
		if(request_mode==0)
			pthread_create( (pthread_t*)&(rwLoopTasksIDs[0][i]), &attr, pipe_task, (Pipe*)&(task_data[0][i]) );
		else if(request_mode==1)
			pthread_create( (pthread_t*)&(rwLoopTasksIDs[1][i]), &attr, pipe_task, (Pipe*)&(task_data[1][i]) );
		else{
			// GET
			pthread_create( (pthread_t*)&(rwLoopTasksIDs[0][i]), &attr, pipe_task, (Pipe*)&(task_data[0][i]) );
			// PUT
			pthread_create( (pthread_t*)&(rwLoopTasksIDs[1][i]), &attr, pipe_task, (Pipe*)&(task_data[1][i]) );
		}
	}
	for(i=0;i<chunk_num;i++){
		pthread_create( (pthread_t*)&(rwLoopTasksIDs[2][i]), &attr, chunk_task, (Chunk*)&(chunk_data[i]) );
	}

	pthread_attr_destroy(&attr);

	for(i=0;i<IO_NUM;i++){
		if(request_mode!=1)	pthread_join(rwLoopTasksIDs[0][i],NULL);
		if(request_mode!=0)	pthread_join(rwLoopTasksIDs[1][i],NULL);
	}
	for(i=0;i<chunk_num;i++){
		pthread_join(rwLoopTasksIDs[2][i],NULL);
	}

	//FREE_END:
	cudaError_t error = cudaDeviceSynchronize();
	time_after = _timestamp();
	double total_time = elapsed_time(time_before,time_after);
	double total_size=0;
	for(i=0;i<IO_NUM;i++){
		total_size+=transfer_size[0][i]+transfer_size[1][i];
	}

	//printf("Transfer time: %lf ms\tCount: %d\tSize: %lf MB\n", totalTime / 1e3,totalCount,(double)totalSize/(1<<20));////1e6);  //ms....
	//printf( "\tBandwidth: %lfGB/s\n\n", ((double)totalSize / (1 << 30)) / (totalTime / 1e6));//1e9));
	printf("SIMPLE | Total time: %lf s\tSize: %lf MB\tBandwidth %lf GB/s \n\n\n", total_time,(total_size/(1<<20)),(total_size/(1<<30))/total_time  );///9));

	//Check for errors and failed asserts in asynchronous kernel launch.

	if (error != cudaSuccess)
	{
		printf("Device failed, CUDA error message is: %s\n\n",
				cudaGetErrorString(error));
	}

	if(call==1){
		PRINT_TIMES("unpinning\n");
		ASSERT_EQ(ioctl(pin_fd,I_KEY_UNPIN_BUF,&unpin),0);
		ASSERTDRV(cuMemFree(gpu_memory));
		close(pin_fd);
	}
	else{
		CUDA_SAFE_CALL(cudaFree(cuda_memory));
		//CUDA_SAFE_CALL(cudaFreeHost(get_pin));
		free(user_get);
		if(request_mode!=0){
			//CUDA_SAFE_CALL(cudaFreeHost(put_pin));
			free(user_put);

		}
	}


	CUDA_SAFE_CALL(cudaFree(cuda_put));
	CUDA_SAFE_CALL(cudaFree(cuda_reset));
	free(request_list);
	for(i=0;i<num_thread;i++){
		pthread_mutex_destroy(&cond_lock[0][i]);
		pthread_mutex_destroy(&cond_lock[1][i]);
		pthread_cond_destroy(&cond[0][i]);
		pthread_cond_destroy(&cond[1][i]);
	}
	for(i=0;i<IO_NUM;i++){
		pthread_cond_destroy(&io_cond[i]);
		pthread_mutex_destroy(&io_lock[i]);
	}

	//	free(temp_data);
	key_close(F_NAME);

	cudaDeviceReset();

	return 0;
}


