#include "gpukv/util.cu.h"
#include "sobel.cu"

__global__ void test_modify(char *data,char *put,int n,unsigned int size){
//	int id=BLOCK_ID;
	int tid,thread_size,num;
	num=NUM_BLOCK;
	tid=TID;
	thread_size=size/NUM_THREADS;

	unsigned long offset;//=BLOCK_ID*size;
	for(int i=BLOCK_ID;i<n;i+=num){ //안되는 경우는 block_ID가 n을 넘겠구나!
		offset=((unsigned long)i*size)+thread_size*tid;
		for(int j=0;j<thread_size;j++){
			put[offset+j]=9-(data[offset+j]-'0')+'0';
		}
		__syncthreads();
	}
	__syncthreads();
}
__global__ void test_app(char *data, char *put,int n,unsigned int size){
	int num;
	unsigned int output_size=size/3;
	num=NUM_BLOCK;

	unsigned long offset,out_offset;//=BLOCK_ID*size;
	for(int i=BLOCK_ID;i<n;i+=num){ //안되는 경우는 block_ID가 n을 넘겠구나!
		offset=((unsigned long)i*size);
		out_offset=((unsigned long)output_size*i);
		sobel((unsigned char*)data+offset,(unsigned char*)put+out_offset);
	}
	__syncthreads();

}
__global__ void test_modify2(char *data,char *put,int *reset,int cnt, int n,unsigned int size){
	int id=BLOCK_ID;
	int tid,thread_size,num;
	num=NUM_BLOCK;
	tid=TID;
	thread_size=size/NUM_THREADS;

	unsigned long offset;//=BLOCK_ID*size;
	for(int i=BLOCK_ID;i<n;i+=num){ //안되는 경우는 block_ID가 n을 넘겠구나!
		offset=((unsigned long)i*size)+thread_size*tid;
		for(int j=0;j<thread_size;j++){
			put[offset+j]=9-(data[offset+j]-'0')+'0';
		}
		__syncthreads();
		if(threadIdx.x+threadIdx.y+threadIdx.z==0){
			reset[num-1-id]=cnt;
			while(readNoCache(&(reset[id]))<cnt);
            printf("%d finish!\n", id);
			//WAIT_ON_MEM(reset[id],cnt);
		}

	}
	__syncthreads();
}

