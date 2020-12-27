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
#include <memory.h>
#include <stdio.h>
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

static void compare_buf(uint32_t *ref_buf, uint32_t *buf, size_t size)
{
    int diff = 0;
    ASSERT_EQ(size % 4, 0U);
    for(unsigned  w = 0; w<size/sizeof(uint32_t); ++w) {
		if (ref_buf[w] != buf[w]) { 
//			if (diff < 10)
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
int main(int argc, char *argv[])
{
    void *dummy;
	int fd,dev_fd,ret;
	struct I_KEY_PIN_PARAM pin;
	struct I_KEY_UNPIN_PARAM unpin;
	struct I_KEY_P2P_PARAM io;

	//ioctl(fd,I_KEY_UNPIN_BUF,&param);
	//ioctl(fd,I_KEY_P2P_GET,&param);
	//ioctl(fd,I_KEY_P2P_PUT,&param);

//	ret=syscall(325,F_NAME); return 0;

    ASSERTRT(cudaMalloc(&dummy, 0));

    const size_t _size = 256*1024+16; //32*1024+8;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;  //bound up 64KB align

    printf("buffer size: %zu\n", size);  //now 320KB
    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));  // device memory allocation
    ASSERTDRV(cuMemsetD8(d_A, 0xA5, size)); //memory set 8bit value  A5로 초기화
    //OUT << "device ptr: " << hex << d_A << dec << endl;

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A)); //we will use p2p sync operation

    uint32_t *init_buf = new uint32_t[size];
    uint32_t *copy_buf = new uint32_t[size];

    init_hbuf_walking_bit(init_buf, size); //이상한 값으로 초기화
    memset(copy_buf, 0xA5, size * sizeof(*copy_buf)); //A5로 초기화   //256KB

	int start_addr=1073741824; //1GB location /lba:262144 hex:40000
	
	dev_fd=open(F_NAME,O_RDWR|O_SYNC);
	if(dev_fd<0){ printf("open error!\n"); return 0;}
	off_t dev_size=lseek(dev_fd,0,SEEK_END);
	if(dev_size<0){
		printf("seek error!\n"); return 0;
	}
	printf("%lfG\n",(double)dev_size/(1024*1024*1024));
	lseek(dev_fd,start_addr,SEEK_SET);
	ret=write(dev_fd,init_buf,size);
	if(ret<0){printf("write error!\n"); return 0;}
	close(dev_fd);

	ret=syscall(325,F_NAME);
	fd=open(D,O_RDWR);
	//ioctl(fd,I_KEY_PIN_BUF,&param);

//    gdr_mh_t mh;
    BEGIN_CHECK {
        CUdeviceptr d_ptr = d_A;
		
		pin.addr=d_A;
		pin.size=size;
		pin.id=0;
		ret=ioctl(fd,I_KEY_PIN_BUF,&pin);
		if(ret!=0){printf("pin error!\n");}


        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled
//        BREAK_IF_NEQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
//        ASSERT_NEQ(mh, 0U);

  //      void *bar_ptr  = NULL;
 //       ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);
        //OUT << "bar_ptr: " << bar_ptr << endl;

 //       gdr_info_t info;
  //      ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
		int off = d_ptr - pin.start_addr;
		cout << "off: " << off << endl;

		io.bar_addr=off;
		io.size=size;
		io.data_offset=start_addr;

		ret=ioctl(fd,I_KEY_P2P_GET,&io);
		if(ret!=0){printf("p2p get error!\n");}
        //uint32_t *buf_ptr = (uint32_t *)((char *)bar_ptr + off);
        //OUT << "buf_ptr:" << buf_ptr << endl;

//        printf("check 1: MMIO CPU initialization + read back via cuMemcpy D->H\n");
//        init_hbuf_walking_bit(buf_ptr, size);
        //mmiowcwb();
        ASSERTDRV(cuMemcpyDtoH(copy_buf, d_ptr, size));
        //ASSERTDRV(cuCtxSynchronize());
        compare_buf(init_buf, copy_buf, size);
        memset(copy_buf, 0xA5, size * sizeof(*copy_buf));
/*
        printf("check 2: gdr_copy_to_bar() + read back via cuMemcpy D->H\n");
        gdr_copy_to_bar(buf_ptr, init_buf, size);
        ASSERTDRV(cuMemcpyDtoH(copy_buf, d_ptr, size));
        //ASSERTDRV(cuCtxSynchronize());
        compare_buf(init_buf, copy_buf, size);
        memset(copy_buf, 0xA5, size * sizeof(*copy_buf));

        printf("check 3: gdr_copy_to_bar() + read back via gdr_copy_from_bar()\n");
        gdr_copy_to_bar(buf_ptr, init_buf, size);
        gdr_copy_from_bar(copy_buf, buf_ptr, size);
        //ASSERTDRV(cuCtxSynchronize());
        compare_buf(init_buf, copy_buf, size);
        memset(copy_buf, 0xA5, size * sizeof(*copy_buf));

        int extra_dwords = 5;
        int extra_off = extra_dwords * sizeof(uint32_t);
        printf("check 4: gdr_copy_to_bar() + read back via gdr_copy_from_bar() + %d dwords offset\n", extra_dwords);
        gdr_copy_to_bar(buf_ptr + extra_dwords, init_buf, size - extra_off);
        gdr_copy_from_bar(copy_buf, buf_ptr + extra_dwords, size - extra_off);
        compare_buf(init_buf, copy_buf, size - extra_off);
        memset(copy_buf, 0xA5, size * sizeof(*copy_buf));

        extra_off = 11;
        printf("check 5: gdr_copy_to_bar() + read back via gdr_copy_from_bar() + %d bytes offset\n", extra_off);
        gdr_copy_to_bar((char*)buf_ptr + extra_off, init_buf, size);
        gdr_copy_from_bar(copy_buf, (char*)buf_ptr + extra_off, size);
        compare_buf(init_buf, copy_buf, size);
*/
  //      printf("unampping\n");
  //      ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
        printf("unpinning\n");
        ASSERT_EQ(ioctl(fd,I_KEY_UNPIN_BUF,&unpin), 0);
    } END_CHECK;
	close(fd);
//    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(cuMemFree(d_A));
    return 0;
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
