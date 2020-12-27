#include <stdio.h>
#include<stdlib.h>
#include <errno.h>

#include "util.cu.h"

#include "key-p2p.h"
#include "key_api.h"

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(err)  __checkCudaErrors (err, __FILE__, __LINE__)
#endif

#include<algorithm>
#include<pthread.h>

#include"original_gpu.cu"

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

pthread_t rwLoopTasksIDs[2][MAX_CHUNK_NUM];

volatile int done = 0;
//pthread_mutex_t rwLoopTasksLocks[RW_HOST_WORKERS];
//pthread_cond_t  rwLoopTasksConds[RW_HOST_WORKERS];
//pthread_t rwLoopTasksIDs[RW_HOST_WORKERS];
//TaskData rwLoopTasksData[RW_HOST_WORKERS];
typedef struct _pipe{
	int id;
	int num_thread;
	int i;
	int chunk_cnt;
	int request_num;
	unsigned int request_size;
	int *request_list;
	char *ptr;
	int call;
}Pipe;
Pipe task_data[2][MAX_CHUNK_NUM];
unsigned long transfer_size[2][MAX_CHUNK_NUM]={0};

volatile int request_status[2][MAX_CHUNK_NUM]={0};
pthread_spinlock_t request_lock[2][MAX_CHUNK_NUM];

pthread_mutex_t cond_lock[2][MAX_CHUNK_NUM];
pthread_cond_t cond[2][MAX_CHUNK_NUM];

void* pipe_task(void* param){
	Pipe* s=(Pipe*)param;
	int request_start[MAX_CHUNK_NUM]={0};
	int id=s->id;
	int chunk_cnt=s->chunk_cnt;
	int request_num=s->request_num;
	int* request_list=s->request_list;
	unsigned int request_size=s->request_size;
	int num_thread=s->num_thread;
	int call=s->call;
	char *buffer=s->ptr;
	unsigned long chunk_size=(unsigned long)request_size*chunk_cnt;
	int next_key=chunk_cnt*num_thread;
	int i,j,start;
	int ret;
	char *temp_buffer;
//	printf("222222222222222222222");
	if(posix_memalign((void**)&temp_buffer,KEY_PAGE_SIZE,request_size)){
		printf("can not allocate io payload buffer!\n");
		return NULL;
	}

	i=s->i; //num_thread;
	//	for(i=0;i<num_thread;i++){
	request_start[i]=i*chunk_cnt;
	//	}
	//	printf("thread %d start!\n",id);
	while(!done){
		//		break;

		int cnt=0,complete_cnt=0;
		//		for(i=0;i<num_thread;i++){
		if(request_start[i]>=request_num){
			complete_cnt++; break; 
		}

		int status;
/*
		pthread_spin_lock(&request_lock[id][i]);
		status=request_status[id][i];
		pthread_spin_unlock(&request_lock[id][i]);

		if((id==0 & status!=0) || (id==1 && status!=1) ){
			pthread_mutex_lock(&cond_lock[id][i]);
			pthread_cond_signal(&cond[id][i]);
			pthread_mutex_unlock(&cond_lock[id][i]);
		}
*/
		status=0; /////////////////////////////////
//		printf("thread_start!\n");
		if(id==0 && status==0){
			cnt++;
			int end=min(request_start[i]+chunk_cnt,request_num);
			start=0;
			for(j=request_start[i];j<end;j++){
				if(call==0){//no p2p
					ret=key_op(KEY_GET,request_list[j],temp_buffer,request_size);
			//		memcpy(buffer+(i*chunk_size)+(start*request_size),temp_buffer,request_size);
//					ret=key_op(KEY_GET,request_list[j],buffer+(i*chunk_size)+(start*request_size),request_size);
				}
				else{
					ret=key_p2p_op(KEY_GET,request_list[j],(i*chunk_size)+(start*request_size),request_size);
				}
				if(ret<=0) printf("key_op error!\n");
				start++;
			}
			transfer_size[id][i]+=(unsigned long)(end-request_start[i])*request_size;

			request_start[i]+=next_key;
			/*
			pthread_spin_lock(&request_lock[id][i]);
			request_status[id][i]=1;
			pthread_spin_unlock(&request_lock[id][i]);
			*/
		}
		else if(id==1 && status==1){
			cnt++;
			int end=min(request_start[i]+chunk_cnt,request_num);

			start=0;
			for(j=request_start[i];j<end;j++){
				memcpy(temp_buffer, buffer+(i*chunk_size)+(start*request_size),request_size);
				ret=key_op(KEY_PUT,START_PUT+request_list[j],temp_buffer,request_size);
				//ret=key_op(KEY_PUT, START_PUT+request_list[j], buffer+(i*chunk_size)+(start*request_size),request_size);
				if(ret<0) printf("key_op error!\n");
				start++;
			}
			transfer_size[id][i]+=(unsigned long)(end-request_start[i])*request_size;

			request_start[i]+=next_key;
			/*
			pthread_spin_lock(&request_lock[id][i]);
			request_status[id][i]=0;
			pthread_spin_unlock(&request_lock[id][i]);
			*/
		}
		//		}
		//		if(complete_cnt>=num_thread) break;

//		if(cnt==0) usleep(100);
	}
	if(id==1){
//		for(i=0;i<num_thread;i++){
		/*
			pthread_spin_lock(&request_lock[id][i]);
			request_status[id][i]=2;
			pthread_spin_unlock(&request_lock[id][i]);
			*/
//		}

	}
	free(temp_buffer);
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

	char *user_get,*user_put;
	user_get=(char*)malloc(sizeof(char)*gpu_memory_size);
	user_put=(char*)malloc(sizeof(char)*gpu_memory_size);
		//	printf("1111111111111\n");
	///////////////////////////////////////////////////////
	///////key_value open
	int key_ret=key_open(F_NAME);

	//	printf("1111111111111\n");
	//////////////////////////////////////////////////////
	//////request list make
	int *request_list;
	int i,j;
	request_list=(int*)malloc(sizeof(int)*request_num);
	for(i=0;i<request_num;i++){
		request_list[i]=i+START_KEY;
	}
	random_shuffle(request_list,request_list+request_num);
//	ASSERTRT(cudaMalloc((void**)&cuda_request,sizeof(int)*request_num));
	//	ASSERTRT(cudaMemcpy(cuda_request,request_list,sizeof(int)*request_num,cudaMemcpyHostToDevice));
	/////////////////////////////////////////////////////
	///////push data to key_value_SSD or GPU_MEMORY
	//	char *temp_data=(char*)malloc(sizeof(char)*gpu_memory_size);
	//	memset(temp_data,0,gpu_memory_size);
	int temp_cnt=0;

	//	printf("1111111111111\n");
	////////ready for START_KEY
	int request_start[MAX_CHUNK_NUM]={0};
	int stage[MAX_CHUNK_NUM]={0};
	int chunk_cnt=(gpu_memory_size/num_thread)/request_size;
	int next_key=chunk_cnt*num_thread;
	unsigned long chunk_size=(unsigned long)chunk_cnt*request_size;
	unsigned long out_size;
	unsigned int out_request_size;

	int n=sqrt(gpu_thread);
	dim3 nthreads(n,n);
	int cuda_block=gpu_block/num_thread;
	double time_before,time_after;

	if(request_mode==3){ out_size=chunk_size/3; out_request_size=request_size/3;} //image
	else{ out_size=chunk_size; out_request_size=request_size;}

	printf("MAX_THROUGHPUT | size : %d KB, num : %d, mode : %d, user_thread : %d, block : %d, thread : %d, p2p_mode : %d\n",
			request_size/1024, request_num,request_mode, num_thread, gpu_block,gpu_thread,call);
	printf("MAX_THROUGHPUT | user_thread : %d, chunk_size : %lf MB, chunk_request_cnt : %d, next_key %d\n",num_thread, (double)chunk_size/(1024*1024),chunk_cnt,next_key);

	for(i=0;i<num_thread;i++){
		request_start[i]=i*chunk_cnt;
	}

	pthread_attr_t attr;
	pthread_attr_init( &attr );
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
	//	return 0;
	//goto FREE_END;
	//	return 0;

	//	return 0;	
	//	pthread_spin_init(&(request_lock[0]),0);
	//	pthread_spin_init(&(request_lock[1]),0);
	//	memset(request_status[0],0,sizeof(int)*MAX_CHUNK_NUM):
	//	memset(request_status[1],0,sizeof(int)*MAX_CHUNK_NUM):

	for(i=0;i<2;i++){
		for(j=0;j<num_thread;j++){
			task_data[i][j].id=i; //for DISK read threads
			task_data[i][j].i=j;
			task_data[i][j].num_thread=num_thread;
			task_data[i][j].chunk_cnt=chunk_cnt;
			task_data[i][j].request_num=request_num;
			if(i==0){
				task_data[i][j].request_size=request_size;
				if(call==0) task_data[i][j].ptr=user_get;
				else task_data[i][j].ptr=NULL;
			}
			else{
				task_data[i][j].request_size=out_request_size;
				task_data[i][j].ptr=user_put;
			}
			task_data[i][j].request_list=request_list;
			task_data[i][j].call=call;
		}
	}

	time_before= _timestamp();

	for(i=0;i<num_thread;i++){
		if(request_mode==0){
			//printf("11111111111");
			pthread_create( (pthread_t*)&(rwLoopTasksIDs[0][i]), &attr, pipe_task, (Pipe*)&(task_data[0][i]) );
		}
		else if(request_mode==1)
			pthread_create( (pthread_t*)&(rwLoopTasksIDs[1][i]), &attr, pipe_task, (Pipe*)&(task_data[1][i]) );
		else{
			//		printf("thread 2!\n");
			pthread_create( (pthread_t*)&(rwLoopTasksIDs[0][i]), &attr, pipe_task, (Pipe*)&(task_data[0][i]) );
			pthread_create( (pthread_t*)&(rwLoopTasksIDs[1][i]), &attr, pipe_task, (Pipe*)&(task_data[1][i]) );
		}
	}

	pthread_attr_destroy(&attr);

//	done=1;
	for(i=0;i<num_thread;i++){
		if(request_mode!=1)	pthread_join(rwLoopTasksIDs[0][i],NULL);
		if(request_mode!=0)	pthread_join(rwLoopTasksIDs[1][i],NULL);
	}

	time_after = _timestamp();
	double total_time = elapsed_time(time_before,time_after);
	double total_size=0;
	for(i=0;i<num_thread;i++){
		total_size+=transfer_size[0][i]+transfer_size[1][i];
	}

	//printf("Transfer time: %lf ms\tCount: %d\tSize: %lf MB\n", totalTime / 1e3,totalCount,(double)totalSize/(1<<20));////1e6);  //ms....
	//printf( "\tBandwidth: %lfGB/s\n\n", ((double)totalSize / (1 << 30)) / (totalTime / 1e6));//1e9));
	printf("MAX_THROUGHPUT | Total time: %lf s\tSize: %lf MB\tBandwidth %lf GB/s \n\n\n", total_time,(total_size/(1<<20)),(total_size/(1<<30))/total_time  );///9));

	//Check for errors and failed asserts in asynchronous kernel launch.
	//////////Success check

	/*	
		if(request_mode==0){
		char temp_value;
		int diff=0,iter=0;
		ASSERTRT(cudaMemcpy(temp_data,cuda_memory,gpu_memory_size,cudaMemcpyDeviceToHost));
		for(i=0;i<gpu_memory_size;i+=request_size){
		temp_value=temp_data[i];
		for(j=i;j<i+request_size;j++){
		if(temp_value!=temp_data[j] && diff<10){ printf("[diff] %c != %c\n",temp_data[j],temp_value); diff++;}
		if(iter<10 && j-i<10) printf("%c",temp_data[j]);
		}
		if(iter<10) printf("\n");
		iter++;
		}
		printf("total diff : %d\n",diff);

		}

		else if(request_mode==1){ //write Check
		memset(buffer[0],0,buffer_size);
		int iter=0,diff=0;
		for(i=START_KEY;i<START_KEY+request_num;i++){
		key_op(GET,i,buffer[0],request_size);
		for(j=0;j<request_size;j++){
		if(buffer[0][j]!=buffer[0][0] && diff<10){ printf("[diff] %c != %c\n",buffer[0][j],buffer[0][0]); diff++;}
		if(iter<10 && j<10) printf("%c",buffer[0][j]);
		}
		if(iter<10) printf("\n");
		iter++;
		}
		printf("total diff : %d\n",diff);
		}

	 */	
	//PRINT_DEBUG;

	///////////////////////////////////////////
	/////All Freeeeee
//	fprintf(stderr, "\n");

	free(user_get);
	free(user_put);
	free(request_list);
	//	free(temp_data);
	key_close(F_NAME);

	cudaDeviceReset();

	return 0;
}


