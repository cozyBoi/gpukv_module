#include "gpukv/fs_calls.cu.h"
#include "gpukv/util.cu.h"

//#define GPU_MEMORY_SIZE (220*1024*1024)
#include "sobel.cu"

#define START_PUT 1000000

__global__ void test_get(char *data,char *put, int *request,int n, unsigned int size, int mode, unsigned int max_size,int num)
{
	int id;
	unsigned long offset;
	if(mode==0 || mode==5){ id=GLOBAL_TID;}// num=NUM_BLOCK*NUM_THREADS;}//thread
	else if(mode==1){ id=GLOBAL_WARP_ID;}// num=NUM_BLOCK*NUM_WARPS; } //warp
	else{ id=BLOCK_ID;} //num=NUM_BLOCK; }//block
	offset=(unsigned long)id*size;
	if(offset+size<=(unsigned long)max_size){
		if(mode==0){ //thread
			for(int i=id;i<n;i+=num)	key_value_op_thread((unsigned long)request[i],offset,size,GET);
		}
		else if(mode==1){ //warp
			for(int i=id;i<n;i+=num)	key_value_op_warp((unsigned long)request[i],offset,size,GET);
		}
		if(mode==2){ //block
			for(int i=id;i<n;i+=num)	key_value_op_block((unsigned long)request[i],offset,size,GET);
		}
		if(mode==3){ //bad_block1
			for(int i=id;i<n;i+=num)	key_value_op_block_bad1((unsigned long)request[i],offset,size,GET);
		}
		if(mode==4){ //bad_block2
			for(int i=id;i<n;i+=num)	key_value_op_block_bad2((unsigned long)request[i],offset,size,GET);
		}
		if(mode==5){ //bad_thread1
			for(int i=id;i<n;i+=num)	key_value_op_thread_bad1((unsigned long)request[i],offset,size,GET);
		}
	}
}
__global__ void test_put(char *data,char *put, int *request,int n, unsigned int size, int mode, unsigned int max_size,int num)
{
	int id;
	unsigned long offset;
	if(mode==0 || mode==5){ id=GLOBAL_TID;}// num=NUM_BLOCK*NUM_THREADS;}//thread
	else if(mode==1){ id=GLOBAL_WARP_ID;}// num=NUM_BLOCK*NUM_WARPS; } //warp
	else{ id=BLOCK_ID;}// num=NUM_BLOCK; }//block
	offset=(unsigned long)id*size;
	if(offset+size<=(unsigned long)max_size){
		if(mode==0){ //thread
			for(int i=id;i<n;i+=num)	key_value_op_thread((unsigned long)request[i],offset,size,PUT);
		}
		else if(mode==1){ //warp
			for(int i=id;i<n;i+=num)	key_value_op_warp((unsigned long)request[i],offset,size,PUT);
		}
		if(mode==2){ //block
			for(int i=id;i<n;i+=num)	key_value_op_block((unsigned long)request[i],offset,size,PUT);
		}
		if(mode==3){ //bad_block1
			for(int i=id;i<n;i+=num)	key_value_op_block_bad1((unsigned long)request[i],offset,size,PUT);
		}
		if(mode==4){ //bad_block2
			for(int i=id;i<n;i+=num)	key_value_op_block_bad2((unsigned long)request[i],offset,size,PUT);
		}
		if(mode==5){ //bad_thread1
			for(int i=id;i<n;i+=num)	key_value_op_thread_bad1((unsigned long)request[i],offset,size,PUT);
		}
	}
}


///////block 개수
__global__ void test_modify(char *data, char *put,int *request, int n, unsigned size,unsigned int max_size,int num){
	int tid,thread_size;
	unsigned long offset;
	int id=BLOCK_ID; 
	offset=(unsigned long)id*size;
	if(offset+size<=(unsigned long)max_size){
//		num=NUM_BLOCK;
		tid=TID;
		thread_size=size/NUM_THREADS;
		for(int i=id;i<n;i+=num){
			key_value_op_block((unsigned long)request[i],offset,size,GET);
			for(int j=0;j<thread_size;j++){
				put[offset+(thread_size*tid)+j]=9-(data[offset+(thread_size*tid)+j]-'0')+'0';
			}
			__syncthreads();
			key_value_op_block((unsigned long)request[i]+START_PUT,offset,size,PUT);
		}
	}
}
__global__ void test_modify2(char *data, char *put,int *request,int *reset, int n, unsigned size,unsigned int max_size,int num){
	int tid,thread_size;
	unsigned long offset;
	int id=BLOCK_ID; 
	int cnt=1;
	offset=(unsigned long)id*size;
	if(offset+size<=(unsigned long)max_size){
//		num=NUM_BLOCK;
		tid=TID;
		thread_size=size/NUM_THREADS;
		for(int i=id;i<n;i+=num){
			int c = data[offset+(thread_size*tid)] - '0';
			key_value_op_block((unsigned long)request[i],offset,size,GET);
			//printf("offset: %d, before: %d  after: %d\n",offset, c, data[offset+(thread_size*tid)] - '0');
			for(int j=0;j<thread_size;j++){
				put[offset+(thread_size*tid)+j]=9-(data[offset+(thread_size*tid)+j]-'0')+'0';
			}
			__syncthreads();
			if(threadIdx.x+threadIdx.y+threadIdx.z==0){
				reset[num-1-id]=cnt;
				while(readNoCache(&(reset[id]))<cnt);
			}
			key_value_op_block((unsigned long)request[i]+START_PUT,offset,size,PUT);
			cnt++;
		}
	}
}

__global__ void test_app(char *data,char *put,int *request,int n, unsigned int size, unsigned int max_size,int num){
//	int num;
	unsigned long offset,out_offset;
	int id=BLOCK_ID; 
	unsigned int out_size=size/3;
	offset=(unsigned long)id*size; out_offset=(unsigned long)id*out_size; //gray_scale
	if(offset+size<=(unsigned long)max_size){
//		num=NUM_BLOCK;
		for(int i=id;i<n;i+=num){
			key_value_op_block((unsigned long)request[i],offset,size,GET);
			sobel((unsigned char*)data+offset,(unsigned char*)put+out_offset);
			key_value_op_block((unsigned long)request[i]+START_PUT,out_offset,out_size,PUT);
		}
	}

}
