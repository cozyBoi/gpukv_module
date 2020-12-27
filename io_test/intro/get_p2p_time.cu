#include <cstdio>
#include <iostream>

#include <stdlib.h>
//#include <sys/time.h>
#include <time.h>

#include <unistd.h>
#include <stdlib.h>
//#include <stdio.h>
#include <fcntl.h>

#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <linux/types.h>
#include <linux/fs.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gpukv/util.cu.h"

#include "key-p2p.h"
#include "key_api.h"


int main(){
    FILE* fp = fopen("../key_log.txt", "r");
    double ac = 0;
	struct timespec st, et;

    int i;
	int device = 0;
	int p2p_size = (192*1024*1024);

	srand(time(NULL));

	//////////// GPU setting
    if(!fp){
        printf("no file!\n");
        return 0;
    }

	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));

	////////////////////p2p memory mapping, and general memory
	void *dummy;
	ASSERTRT(cudaMalloc(&dummy,0));

	CUdeviceptr gpu_memory;
	char* cuda_memory;

	ASSERTDRV(cuMemAlloc(&gpu_memory, p2p_size));
	ASSERTDRV(cuMemsetD8(gpu_memory, 0x00, p2p_size));

	unsigned int flag=1;
	ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, gpu_memory));

	struct I_KEY_PIN_PARAM pin;
	struct I_KEY_UNPIN_PARAM unpin;
	int pin_fd;

	pin_fd=open(D,O_RDWR);
	pin.addr=gpu_memory;
	pin.size=p2p_size;
	pin.id=0;
	int ret=ioctl(pin_fd,I_KEY_PIN_BUF,&pin);
	if(ret!=0){printf("gpu pin error!\n"); return 0;}
    
    cuda_memory=(char*)gpu_memory;

	////////// KV open
	int key_ret=key_open(F_NAME);
	int buffer_size=KEY_PAGE_SIZE;
	char *buffer[10];
	for(i = 0; i < 10; ++i){
		if(posix_memalign((void**)&buffer[i],KEY_PAGE_SIZE,buffer_size)){
			printf("can not allocate io payload buffer!\n");
			return 0;
		}
	}


    //struct timespec accum={0};
	clock_gettime(CLOCK_MONOTONIC, &st);
	for(int i = 0; i < 1957027; i++){
        int key;
		struct timespec start, end, diff;
        fscanf(fp, "%d", &key);

		clock_gettime(CLOCK_MONOTONIC, &start);
        key_p2p_op(KEY_GET, key, (i%49152)*4096, 4096);
		clock_gettime(CLOCK_MONOTONIC, &end);

        ac += (1000000000LL * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec))/1000.0f;
	}
	clock_gettime(CLOCK_MONOTONIC, &et);

    printf("get time(w/o read data): %f us\n", ac/1957057);
    printf("In real: %lf us\n", (1000000000LL*(et.tv_sec - st.tv_sec) + (et.tv_nsec - st.tv_nsec))/1957057/1000.0f);


	ASSERT_EQ(ioctl(pin_fd,I_KEY_UNPIN_BUF,&unpin),0);

	ASSERTDRV(cuMemFree(gpu_memory));

	close(pin_fd);
	key_close(F_NAME);

	cudaDeviceReset();


    return 0;
}
