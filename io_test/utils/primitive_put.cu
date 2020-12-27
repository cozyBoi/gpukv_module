#include<stdio.h>
#include<stdlib.h>
#include<errno.h>
#include<math.h>
#include<assert.h>
#include<algorithm>
#include <random>

#include"key_api.h"
#include"key-p2p.h"

#include "gpukv/timer.h"
#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(err)  __checkCudaErrors (err, __FILE__, __LINE__)
#endif

#define TOTAL_VERTICES 1971279

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int) err, cudaGetErrorString(err));
		exit(-1);
	}
}

int main(int argc, char** argv)
{
	int device = 0;
	int request_num=1024, request_mode=1;
	unsigned int request_size=4096;
	int gpu_memory_size=(220*1024*1024); ///p2p mem :220MB
	int start_key=0,num_thread;
//	int start_key = 0, num_thread;

	int ret=0;
	double total_size;

	while(1){
		int c;
		c=getopt(argc,argv,"s:n:m:p:i:b:t:c:q:h");
		if(c==-1) break;
		switch(c){
			case 'k':
				start_key=strtol(optarg,NULL,0);
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
			case 'h':
				printf("syntax: %s -k <start_key> -s <request_size> -n <num_request> -m <mode:get-0,put-1> -p <num_polling_thread>  -h\n",argv[0]);
				exit(1);
				break;
			default:
				printf("ERROR: invalid option\n");
				exit(1);
				break;
		}
	}

	int *request_list,*cuda_request;
	int i,j,cnt;
	request_list=(int*)malloc(sizeof(int)*request_num);
	for(i=0;i<request_num;i++){
		request_list[i]=i+start_key;
	}
	random_shuffle(request_list,request_list+request_num);
	
    /*
	int buffer_size=KEY_PAGE_SIZE*1024;
	char *buffer[10];
	int temp_cnt=0;
	for(i=0;i<10;i++){
		if(posix_memalign((void**)&buffer[i],KEY_PAGE_SIZE,buffer_size)){
			printf("can not allocate io payload buffer!\n");
			return 0;
		}
		memset(buffer[i],i+'0',buffer_size);
	}
    */
    int buffer_size = KEY_PAGE_SIZE;
    unsigned char *buffer;
    if(posix_memalign((void**)&buffer, KEY_PAGE_SIZE, buffer_size)){
        printf("can not allocate io payload buffer!\n");
        return 0;
    }

	total_size=(double)((double)request_size*request_num)/(1024*1024);

	printf("PUSH num : %d, size : %d, total : %lf MB\n",request_num,request_size,total_size);

	//int key_ret=key_open(F_NAME);

	double time_before = _timestamp();
	unsigned long key;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<unsigned char> distribution(0, 255);


#pragma omp parallel for num_threads(64)
	for(i=0;i<request_num;i++){
        for(int j = 0; j < KEY_PAGE_SIZE; ++j){
            buffer[j] = distribution(generator);
        }
        char fname[100];
        FILE* fp;

        key = i;
        sprintf(fname, "dataset/synthetic/%d.txt", key);
		//cnt=key%10;
        fp = fopen(fname, "w");
        fwrite(buffer, sizeof(char), request_size, fp);
        //fwrite(buffer[cnt], sizeof(char), request_size, fp);
		//ret=key_op(KEY_PUT,key,buffer[cnt],request_size);
        fclose(fp);
		//if(ret!=request_size) printf("error! ret=%d\n",ret);
	}
	double time_after = _timestamp();
	double total_time=elapsed_time(time_before,time_after);
	fprintf(stderr, "first | total time: %lf s, Bandwidth %lf GB/s \n",total_time,(double)((total_size)/(1<<10))/total_time);///9));
	printf("END PUSH|  num : %d, size : %d, total : %lf MB\n",request_num,request_size,total_size);

	//key_close(F_NAME);
	return 0;
}
	/*	 
		for(i=0;i<10;i++){
			for(j=0;j<KEY_PAGE_SIZE*256;j++) temp_ptr[j]=temp_data[i*4096+j];
			ret=key_op(PUT,i+10000,temp_ptr,KEY_PAGE_SIZE*256);
			printf("put ret :%d\n",ret);
		}*/

		/*
		for(i=0;i<10;i++){
			ret=key_op(GET,i+10000,temp_ptr,4096);
			printf("get ret :%d\n",ret);
			for(j=0;j<10;j++){
				printf("%c",temp_ptr[j]);
			}
			printf("\n");
		}
		free(buffer);
		*/
		///////////////////////////////////////////////////
		///put p2p test
		/*
		for(i=0;i<10;i++){
			ret=key_p2p_op(PUT,i+10000,i*KEY_PAGE_SIZE,KEY_PAGE_SIZE);
			printf("PUT_p2p ret :%d\n",ret);
		}
		memset(temp_ptr,0,sizeof(char)*buffer_size);
		for(i=0;i<10;i++){
			ret=key_op(GET,i+10000,temp_ptr,KEY_PAGE_SIZE);
			printf("get ret :%d\n",ret);
			for(j=0;j<10;j++){
				printf("%c",temp_ptr[j]);
			}
			printf("\n");
		}*/
		//get_p2p_test
		/*
		ASSERTDRV(cuMemsetD8(gpu_memory, 0xA5, gpu_memory_size)); //memory set 8bit value  A5로 초기화
		memset(temp_data,0,sizeof(char)*gpu_memory_size);
		/////////////////////////////p2p put test_get
		for(i=0;i<10;i++){
			ret=key_p2p_op(GET,i+10000,i*KEY_PAGE_SIZE*256,KEY_PAGE_SIZE*256);
			printf("get_p2p ret :%d\n",ret);
		}
		ASSERTDRV(cuMemcpyDtoH(temp_data,gpu_memory,gpu_memory_size));
//		printf("init_test: ");
//		for(j=0;j<10;j++)	printf("%c",temp_data[j]);
//		printf("\n");
		for(i=0;i<10;i++){
//			ret=key_op(GET,i+10000,temp_ptr,4096);
//			printf("get ret :%d\n",ret);
			for(j=0;j<10;j++){
				printf("%c",temp_data[j+i*KEY_PAGE_SIZE*256]);
			}
			printf("\n");
		}
		*/


		/*
		*/
//		free(buffer);
//		free(temp_data);
//	return 0;

