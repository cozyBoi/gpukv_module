/*
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <getopt.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <sys/types.h>
#include <asm/types.h>
#include <sys/ioctl.h>

using namespace std;


#include "key-p2p.h"

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    ((__u64)1 << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

#define F_NAME "/dev/nvme0n1"
#define D "/dev/key_value_module"

//#pragma once

#define ASSERT(x)                                                       \
	do                                                                  \
{                                                               \
	if (!(x))                                                   \
	{                                                       \
		fprintf(stdout, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__, __LINE__); \
		exit(EXIT_FAILURE);                                 \
	}                                                       \
} while (0)

#define ASSERTDRV(stmt)				\
	do                                          \
{                                       \
	CUresult result = (stmt);           \
	ASSERT(CUDA_SUCCESS == result);     \
} while (0)

#define ASSERTRT(stmt)				\
	do                                          \
{                                       \
	cudaError_t result = (stmt);           \
	ASSERT(cudaSuccess == result);     \
} while (0)

#define ASSERT_EQ(P, V) ASSERT((P) == (V))
#define CHECK_EQ(P, V) ASSERT((P) == (V))
#define ASSERT_NEQ(P, V) ASSERT((P) != (V))
#define BREAK_IF_NEQ(P, V) if((P) != (V)) break
#define BEGIN_CHECK do
#define END_CHECK while(0)


#define OUT cout
//#define OUT TESTSTACK

//#define MYCLOCK CLOCK_REALTIME
//#define MYCLOCK CLOCK_RAW_MONOTONIC
#define MYCLOCK CLOCK_MONOTONIC
static void compare_buf(uint32_t *ref_buf, uint32_t *buf, size_t size)
{
	int diff = 0;
	ASSERT_EQ(size % 4, 0U);
	for(unsigned  w = 0; w<size/sizeof(uint32_t); ++w) {
		if (ref_buf[w] != buf[w]) { 
			if (diff < 10)
				printf("[word %d] %08x != %08x\n", w, buf[w], ref_buf[w]);
			++diff;
		}
	}
	//OUT << "diff(s): " << diff << endl;
	//CHECK_EQ(diff, 0);
	if (diff) {
		cout << "check error: diff(s)=" << diff << endl;
	}
}

static void init_hbuf_walking_bit(uint32_t *h_buf, size_t size)
{
	uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
	unsigned w;
	ASSERT_NEQ(h_buf, (void*)0);
	ASSERT_EQ(size % 4, 0U);
	//OUT << "filling mem with walking bit " << endl;
	for(w = 0; w<size/sizeof(uint32_t); ++w)
		h_buf[w] = base_value ^ (1<< (w%32));
}

static void init_hbuf_linear_ramp(uint32_t *h_buf, size_t size)
{
	uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
	unsigned w;
	ASSERT_NEQ(h_buf, (void*)0);
	ASSERT_EQ(size % 4, 0U);
	//OUT << "filling mem with walking bit " << endl;
	for(w = 0; w<size/sizeof(uint32_t); ++w)
		h_buf[w] = w;
}
#define LBA_SIZE 4096

// manually tuned...
int num_write_iters = 100;
int num_read_iters  = 100;

int main(int argc, char *argv[])
{
	size_t _size = 64*1024*1024;
	size_t copy_size = 0;
	size_t copy_offset = 0;
	int dev_id = 0;

	while(1) {        
		int c;
		c = getopt(argc, argv, "s:d:o:c:w:r:h");
		if (c == -1)
			break;

		switch (c) {
			case 's':
				_size = strtol(optarg, NULL, 0);
				break;
			case 'c':
				copy_size = strtol(optarg, NULL, 0);
				break;
			case 'o':
				copy_offset = strtol(optarg, NULL, 0);
				break;
			case 'd':
				dev_id = strtol(optarg, NULL, 0);
				break;
			case 'w':
				num_write_iters = strtol(optarg, NULL, 0);
				break;
			case 'r':
				num_read_iters = strtol(optarg, NULL, 0);
				break;
			case 'h':
				printf("syntax: %s -s <buf size> -c <copy size> -o <copy offset> -d <gpu dev id> -w <write iters> -r <read iters> -h\n", argv[0]);
				exit(EXIT_FAILURE);
				break;
			default:
				printf("ERROR: invalid option\n");
				exit(EXIT_FAILURE);
		}
	}
	copy_size*=1024;
	if (!copy_size)
		copy_size = _size;

	if (copy_offset % sizeof(uint32_t) != 0) {
		printf("ERROR: offset must be multiple of 4 bytes\n");
		exit(EXIT_FAILURE);
	}

	if (copy_offset + copy_size > _size) {
		printf("ERROR: offset + copy size run past the end of the buffer\n");
		exit(EXIT_FAILURE);
	}

	size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

	int n_devices = 0;
	ASSERTRT(cudaGetDeviceCount(&n_devices));

	cudaDeviceProp prop;
	for (int n=0; n<n_devices; ++n) {
		ASSERTRT(cudaGetDeviceProperties(&prop,n));
		OUT << "GPU id:" << n << " name:" << prop.name 
			<< " PCI domain: " << prop.pciDomainID 
			<< " bus: " << prop.pciBusID 
			<< " device: " << prop.pciDeviceID << endl;
	}
	OUT << "selecting device " << dev_id << endl;
	ASSERTRT(cudaSetDevice(dev_id));

	void *dummy;
	int fd,dev_fd,ret;
	struct I_KEY_PIN_PARAM pin;
	struct I_KEY_UNPIN_PARAM unpin;
	struct I_KEY_P2P_PARAM io;

	ASSERTRT(cudaMalloc(&dummy, 0));

	OUT << "testing size: " << _size << endl;
	OUT << "rounded size: " << size << endl;

	CUdeviceptr d_A;
	ASSERTDRV(cuMemAlloc(&d_A, size));
	OUT << "device ptr: " << hex << d_A << dec << endl;

	unsigned int flag = 1;
	ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

	uint32_t *init_buf = NULL;
	uint32_t *copy_buf = NULL;
	init_buf = (uint32_t *)malloc(size);
	copy_buf = (uint32_t *)malloc(size);
	ASSERT_NEQ(init_buf, (void*)0);
	ASSERT_NEQ(copy_buf, (void*)0);
	init_hbuf_walking_bit(init_buf, size);
    memset(copy_buf, 0xA5, size ); //A5로 초기화


	int start_addr=1073741824; //1GB location lba:2097152, hex20 0000

	dev_fd=open(F_NAME,O_RDWR|O_SYNC);
	if(dev_fd<0){ printf("open error!\n"); return 0;}
	off_t dev_size=lseek(dev_fd,0,SEEK_END);
	if(dev_size<0){
		printf("seek error!\n"); return 0;
	}
	printf("%lfG\n",(double)dev_size/(1024*1024*1024));
	lseek(dev_fd,start_addr,SEEK_SET);
	ret=write(dev_fd,init_buf,size);		//device에 init_buf write

	printf("I'will be sleep 10\n");
	sleep(10);
	if(ret<0){printf("write error!\n"); return 0;}
	close(dev_fd);

	ret=syscall(325,F_NAME);
	fd=open(D,O_RDWR);


	//gdr_t g = gdr_open();
	//ASSERT_NEQ(g, (void*)0);

	//    gdr_mh_t mh;
	BEGIN_CHECK {
		// tokens are optional in CUDA 6.0
		// wave out the test if GPUDirectRDMA is not enabled

		pin.addr=d_A;
		pin.size=size;
		pin.id=0;
		ret=ioctl(fd,I_KEY_PIN_BUF,&pin);		//pinning buffer
		printf("I'will be sleep 10\n");
		sleep(10);

		if(ret!=0){printf("pin error!\n");}

		//BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, size, 0, 0, &mh), 0);
		//ASSERT_NEQ(mh, 0U);

		//void *bar_ptr  = NULL;
		//ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);
		//OUT << "bar_ptr: " << bar_ptr << endl;

		//gdr_info_t info;
		//ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
		OUT << "info.va: " << hex << pin.start_addr << dec << endl;
		OUT << "info.mapped_size: " << pin.pin_size << endl;
		//      OUT << "info.page_size: " << info.page_size << endl;

		// remember that mappings start on a 64KB boundary, so let's
		// calculate the offset from the head of the mapping to the
		// beginning of the buffer
		int off = pin.start_addr-d_A;//info.va - d_A;
		OUT << "page offset: " << off << endl;

		io.bar_addr=off;
		io.size=copy_size;
		io.data_offset=start_addr;
		//uint32_t *buf_ptr = (uint32_t *)((char *)bar_ptr + off);
		// OUT << "user-space pointer:" << buf_ptr << endl;

		// copy to BAR benchmark
		cout << "BAR writing test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_write_iters << endl;
		struct timespec beg, end;
		if(copy_size<64*1024){
			clock_gettime(MYCLOCK, &beg);
			for(int iter=0;iter<num_write_iters;++iter){
				io.bar_addr=off+iter*(64*1024);
				io.data_offset=start_addr+iter*copy_size;
				ret=ioctl(fd,I_KEY_P2P_GET,&io);		//p2p read request (SSD->GPU)
				if(ret!=0){printf("p2p get error!\n");}
			}
		}
		else{
			clock_gettime(MYCLOCK, &beg);
			for (int iter=0; iter<num_write_iters; ++iter){
				io.bar_addr=off+iter*copy_size;
				io.data_offset=start_addr+iter*copy_size;
				ret=ioctl(fd,I_KEY_P2P_GET,&io);		//p2p read request (SSD->GPU)
				if(ret!=0){printf("p2p get error!\n");}
		//            gdr_copy_to_bar(buf_ptr + copy_offset/4, init_buf, copy_size);
			}
		}
		//	if(ret!=0){printf("p2p get error!\n");}
		//            gdr_copy_to_bar(buf_ptr + copy_offset/4, init_buf, copy_size);
		clock_gettime(MYCLOCK, &end);

		double woMBps;
		{
			double byte_count = (double) copy_size * num_write_iters;
			double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
			double Bps = byte_count / dt_ms * 1e3;
			woMBps = Bps / 1024.0 / 1024.0;
			cout << "BAR1 write BW: " << woMBps << "MB/s" << endl;
		}
		
		int num=copy_size*num_write_iters;
		if(copy_size>=64*1024){
		ASSERTDRV(cuMemcpyDtoH(copy_buf, d_A, num)); //GPU data read GPU-> CPU
		printf("1");
		compare_buf(init_buf, copy_buf, num);
		printf("2");
		memset(copy_buf, 0xA5, size);
		printf("3");
		}
/*
		// copy from BAR benchmark
		cout << "BAR reading test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_read_iters << endl;
		clock_gettime(MYCLOCK, &beg);
		for (int iter=0; iter<num_read_iters; ++iter)
			ret=ioctl(fd,I_KEY_P2P_PUT,&io);		//p2p read request (SSD->GPU)
		// gdr_copy_from_bar(init_buf, buf_ptr + copy_offset/4, copy_size);
		clock_gettime(MYCLOCK, &end);

		double roMBps;
		{
			double byte_count = (double) copy_size * num_read_iters;
			double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
			double Bps = byte_count / dt_ms * 1e3;
			roMBps = Bps / 1024.0 / 1024.0;
			cout << "BAR1 read BW: " << roMBps << "MB/s" << endl;
		}
*/
		printf("unpinning\n");
		ASSERT_EQ(ioctl(fd,I_KEY_UNPIN_BUF,&unpin), 0);
		//     OUT << "unmapping buffer" << endl;
		//    ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);

		//  OUT << "unpinning buffer" << endl;
		// ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
	} END_CHECK;

	//  OUT << "closing gdrdrv" << endl;
	// ASSERT_EQ(gdr_close(g), 0);

	ASSERTDRV(cuMemFree(d_A));
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
