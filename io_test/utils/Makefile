ROOTDIR?=../..
CUDA_ROOT:=/usr/local/cuda
ROCKS_CONFIG:=$(ROOTDIR)/rocksdb/make_config.mk

include $(ROCKS_CONFIG)

INCLUDEDIR=$(ROOTDIR)/include
LIBDIR=$(ROOTDIR)/lib
SHARED_LIBS=-lgpufs -lrocksdb -lpthread -ldl -lgomp
ROCKS_SHARED_LIBS=$(SHARED_LIBS) -lz -lrt
CUDA_SHARED_LIBS=$(SHARED_LIBS) -lcudart -lcuda


CUDA_INCLUDE=-I$(INCLUDEDIR) -I$(CUDA_ROOT)/include
CUDA_LIB=-L$(LIBDIR) -L$(CUDA_ROOT)/lib64 -L $(CUDA_ROOT)/lib -L /usr/lib64/nvidia -L /usr/lib/nvidia $(CUDA_SHARED_LIBS)

CC=/usr/bin/gcc
CXX=/usr/bin/g++

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

CXXFLAGS=-L$(LIBDIR) $(ROCKS_SHARED_LIBS) -I$(INCLUDEDIR) --std=c++11
NVCCFLAGS=-g -Xcompiler -fopenmp -G -O2  $(CUDA_LIB) $(CUDA_INCLUDE) $(EXTRA_FLAGS) --std=c++11 --generate-code code=sm_61,arch=compute_61 -maxrregcount 32

ROCKS_FLAGS=-D DATA_ON_ROCKS

PLATFORM_CXXFLAGS= -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX  -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DZLIB -DNUMA -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DHAVE_SSE42  -DHAVE_PCLMUL  -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL



BIN=../bin/utils

all: graph_put graph_put_rocks primitive_put graph_get

graph_put.o: graph_put.cu
	$(NVCC) $(NVCCFLAGS) -g -dc $< -o $@

graph_put_rocks.o: graph_put.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(ROCKS_FLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS) -g -dc

primitive_put.o: primitive_put.cu
	$(NVCC) $(NVCCFLAGS) -g -dc $< -o $@

primitive_put_rocks.o: primitive_put.cu
	$(NVCC) $(ROCKS_FLAGS) $(NVCCFLAGS) $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -g -dc $< -o $@ -lz -lrt


graph_put: graph_put.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" $<
	mv $@ $(BIN)

graph_put_rocks: graph_put_rocks.o | $(BIN)
	$(CXX) -o "$@" $< $(CXXFLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS)
	mv $@ $(BIN)

primitive_put: primitive_put.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" $<
	mv $@ $(BIN)

primitive_put_rocks: primitive_put_rocks.o | $(BIN)
	$(NVCC) $(PLATFORM_LDFLAGS) $(PLAGFORM_CXXFLAGS) $(EXEC_LDFLAGS) $(NVCCFLAGS) -g --cudart static -link -o "$@" $< -lz -lrt
	mv $@ $(BIN)


graph_get: graph_get.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" $<
	mv $@ $(BIN)
graph_get.o: graph_get.cu
	$(NVCC) $(NVCCFLAGS) -g -dc $< -o $@


$(BIN):
	mkdir -p $(BIN)

clean:
	rm -f *.o
