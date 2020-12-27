ROOTDIR?=../..
CUDA_ROOT:=/usr/local/cuda
ROCKS_CONFIG:=$(ROOTDIR)/rocksdb/make_config.mk

include $(ROCKS_CONFIG)

INCLUDEDIR=$(ROOTDIR)/include
LIBDIR=$(ROOTDIR)/lib
SHARED_LIBS=-lgpufs -lrocksdb -lcudart -lcuda -lpthread -ldl


CUDA_INCLUDE=-I$(INCLUDEDIR) -I$(CUDA_ROOT)/include
CUDA_LIB=-L$(LIBDIR) -L$(CUDA_ROOT)/lib64 -L $(CUDA_ROOT)/lib -L /usr/lib64/nvidia -L /usr/lib/nvidia $(SHARED_LIBS)

CC=/usr/bin/g++

ifeq ($(t),3) #thread
	EXTRA_FLAGS=-DRW_IPC_SIZE=28672
else
ifeq ($(t),2)
	EXTRA_FLAGS=-DRW_IPC_SIZE=1024
else
ifeq ($(t),4)
	EXTRA_FLAGS=-DRW_IPC_SIZE=$d
else   #block
	EXTRA_FLAGS=-DRW_IPC_SIZE=256
endif
endif
endif

ifeq ($(l),1) #timing
	EXTRA_FLAGS+=-DMALLOC_STATS -DTIMING_STATS -DTRACE #malloc,timing,trace
endif

NVCC=$(CUDA_ROOT)/bin/nvcc

CXXFLAGS=-O2 $(CUDA_LIB) $(CUDA_INCLUDE) $(EXTRA_FLAGS) --std=c++11
NVCCFLAGS=-g -G -O2 $(CUDA_LIB) $(CUDA_INCLUDE) $(EXTRA_FLAGS) --std=c++11 --generate-code code=sm_61,arch=compute_61 -maxrregcount 32

ROCKS_FLAGS=-D DATA_ON_ROCKS

PLATFORM_CXXFLAGS= -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX  -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DZLIB -DNUMA -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DHAVE_SSE42  -DHAVE_PCLMUL  -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL



BIN=../bin/tests

all: rocks_test

rocks_test.o: rocks_test.cu
	$(NVCC) $(ROCKS_FLAGS) $(NVCCFLAGS) $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -g -dc $< -o $@ -lz -lrt

rocks_test: rocks_test.o | $(BIN)
	$(NVCC) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS) $(NVCCFLAGS) -g --cudart static -link -o "$@" $< -lz -lrt
	mv $@ $(BIN)

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -f *.o
