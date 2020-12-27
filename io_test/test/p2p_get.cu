#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gpukv/util.cu.h"

#include "key-p2p.h"
#include "key_api.h"

// gardenia
#include "gardenia/timer.h"

int main(int argc, char **argv){
	int i;
	int device = 0;
	int p2p_size = (192*1024*1024);

	srand(time(NULL));

	//////////// GPU setting

	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	//PRINT_TIMES("GPU id: %d, name: %s\nPCI domain: %d, bus: %d, device: %d\n",device,deviceProp.name,deviceProp.pciDomainID,deviceProp.pciBusID,deviceProp.pciDeviceID);

	//PRINT_TIMES("Running on device %d: \"%s\"\n", device, deviceProp.name);

	//PRINT_TIMES("\tblocks %d threads %d\n", gpu_block, gpu_thread);
	////////////////////////////////////////////////
	////////////////////p2p memory mapping, and general memory
	void *dummy;
	ASSERTRT(cudaMalloc(&dummy,0));

	CUdeviceptr gpu_memory;
	char* cuda_memory;

	ASSERTDRV(cuMemAlloc(&gpu_memory, p2p_size));
	ASSERTDRV(cuMemsetD8(gpu_memory, 0x00, p2p_size));
	//PRINT_TIMES("device ptr: 0x%lx \n",gpu_memory);
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
	//printf("[DEBUG:gpukv_bfs] pinned!\n");
	//PRINT_TIMES("info.va: %lx\n",pin.start_addr);   
	//PRINT_TIMES("info.mapped_size: %lu B,(%lu MB)\n",pin.pin_size,pin.pin_size/(1024*1024));
	//      OUT << "info.page_size: " << info.page_size << endl;

	// remember that mappings start on a 64KB boundary, so let's
	// calculate the offset from the head of the mapping to the
	// beginning of the buffer
	//int start_off = pin.start_addr-gpu_memory;//info.va - d_A;
	//PRINT_TIMES("page offset: %lx\n", start_off);
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

	//////////// PUT data to SSD
	/*
	for(i = 0; i < 25600; i++){ // write 100MB
		char num = rand() % 10 + '0';
		memset(buffer[i%10], num, buffer_size);
		key_op(KEY_PUT, i, buffer[i % 10], buffer_size);
	}
	
	printf("data setup at ssd finished : 1000 keys with each 4KB values\n");
	*/
	/////////////////////// Calculate P2P Time
	/**/
	Timer t, t1;
	t.Start();
	for(i = 0; i < 1000; i+=4){
		t1.Start();
		//ret=key_p2p_op(KEY_GET, i, i*4096, 4096);
		ret = key_op(KEY_GET, i, buffer[i%10], buffer_size);
		t1.Stop();
		printf("[iter:%d] P2P time: %f us.\n", i, t1.Microsecs());
		//if(i == 0) getchar();
	}
	t.Stop();
	printf("Total: %f us.\n", t.Microsecs());
	//*/
	// finish
	ASSERT_EQ(ioctl(pin_fd,I_KEY_UNPIN_BUF,&unpin),0);



	ASSERTDRV(cuMemFree(gpu_memory));

	close(pin_fd);
	key_close(F_NAME);

	cudaDeviceReset();

	return 0;
}
