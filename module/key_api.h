#ifndef KEY_VALUE_API
#define KEY_VALUE_API

#include<linux/unistd.h>
#include<string.h>
#include<stdlib.h>

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

#define __key_open 326
#define __key_close 327
#define __key_op 323
#define __key_p2p_op 324
#define __key_batch_op 325

#define KEY_GET 0x02
#define KEY_PUT 0x01
//#define KEY_PUT 0xA0
//#define KEY_GET 0xA1
#define KEY_PAGE_SIZE 4096
//#define KEY_DEFAULT_BUF (KEY_PAGE_SIZE*256)

int key_open(char *buf){
	return syscall(__key_open,buf);
}
int key_close(char *buf){
	return syscall(__key_close,buf);
}

int key_op(int mode, unsigned long key, char* buf, unsigned int length){
	return syscall(__key_op,mode,key,buf,length);
}
int key_p2p_op(int mode, unsigned long key,unsigned long offset,unsigned int length){
	return syscall(__key_p2p_op,mode,key,offset,length);
}
int key_batch_op(unsigned long arg,int len,int mode){
	return syscall(__key_batch_op,arg,len,mode);
}

////////////////
#define F_NAME "/dev/nvme0n1"
#define D "/dev/key_value_module"

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



#endif
