ROOT_PATH?=../..
ROCKS_CONFIG:=$(ROOT_PATH)/rocksdb/make_config.mk
include $(ROCKS_CONFIG)

CUDA_ROOT:=/usr/local/cuda

INCLUDEDIR=$(ROOT_PATH)/include
LIBDIR=$(ROOT_PATH)/lib
SHARED_LIBS=-lgpufs -lpthread -ldl -lgomp
ROCKS_SHARED_LIBS=-lrocksdb -lz -lrt
CUDA_SHARED_LIBS=-lcudart -lcuda -lcurand

CUDA_INCLUDE=-I$(INCLUDEDIR) -I$(CUDA_ROOT)/include
CUDA_LIB=-L$(LIBDIR) -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib -L /usr/lib64/nvidia -L /usr/lib/nvidia
CUDA_APP_LIB=$(CUDA_LIB) $(SHARED_LIBS) $(CUDA_SHARED_LIBS)

BIN=../bin/synthetic

CC=/usr/bin/g++
NVCC=$(CUDA_ROOT)/bin/nvcc

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

CXXFLAGS=-L$(LIBDIR) $(SHARED_LIBS) $(ROCKS_SHARED_LIBS) -I$(INCLUDEDIR) --std=c++11
NVCC_EXTRA_FLAGS=--std=c++11 --generate-code code=sm_61,arch=compute_61 -maxrregcount 32
NVCCFLAGS=-g -Xcompiler -fopenmp -G -O2 $(CUDA_APP_LIB) $(CUDA_INCLUDE) $(EXTRA_FLAGS) $(NVCC_EXTRA_FLAGS) # -DDEBUG -DTIMING_STATS -DTRACE

PLATFORM_CXXFLAGS= -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX  -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DZLIB -DNUMA -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DHAVE_SSE42  -DHAVE_PCLMUL  -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL

ROCKS_EXTRA_FLAGS=$(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

all: gpukv conv_kvssd conv_rocks_direct conv_rocks_default conv_rocks_cpp_direct conv_rocks_cpp_default 

kernel.o: kernel.cu
	$(NVCC) $(NVCCFLAGS) -g -dc -c $<
gpukv.o: gpukv.cu
	$(NVCC) $(NVCCFLAGS) -g -dc -c $<
conv_kvssd.o: conv.cu
	$(NVCC) $(NVCCFLAGS) -g -dc -c $< -o $@
conv_rocks_default.o: conv.cu
	$(NVCC) $(NVCCFLAGS) $(ROCKS_SHARED_LIBS) $(ROCKS_EXTRA_FLAGS) -D DATA_ON_ROCKS -g -dc -c $< -o $@ -lz -lrt
conv_rocks_direct.o: conv.cu
	$(NVCC) $(NVCCFLAGS) $(ROCKS_SHARED_LIBS) $(ROCKS_EXTRA_FLAGS) -D DATA_ON_ROCKS -D DIRECTIO -g -dc -c $< -o $@ -lz -lrt
conv_rocks_cpp_default.o: conv.cu
	$(NVCC) $(NVCCFLAGS) $(ROCKS_SHARED_LIBS) $(ROCKS_EXTRA_FLAGS) -D DATA_ON_ROCKS -D ROCKS_CXX -g -dc -c $< -o $@ -lz -lrt
conv_rocks_cpp_direct.o: conv.cu
	$(NVCC) $(NVCCFLAGS) $(ROCKS_SHARED_LIBS) $(ROCKS_EXTRA_FLAGS) -D DATA_ON_ROCKS -D ROCKS_CXX -D DIRECTIO -g -dc -c $< -o $@ -lz -lrt




gpukv: kernel.o gpukv.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" gpukv.o kernel.o
	mv $@ $(BIN)
conv_kvssd: kernel.o conv_kvssd.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" conv_kvssd.o kernel.o
	mv $@ $(BIN)
conv_rocks_default: kernel.o conv_rocks_default.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ROCKS_SHARED_LIBS) $(ROCKS_EXTRA_FLAGS) -g --cudart static -link -o "$@" kernel.o conv_rocks_default.o
	mv $@ $(BIN)
conv_rocks_direct: kernel.o conv_rocks_direct.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ROCKS_SHARED_LIBS) $(ROCKS_EXTRA_FLAGS) -g --cudart static -link -o "$@" kernel.o conv_rocks_direct.o
	mv $@ $(BIN)

conv_rocks_cpp_default: kernel.o conv_rocks_cpp_default.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ROCKS_SHARED_LIBS) $(ROCKS_EXTRA_FLAGS) -g --cudart static -link -o "$@" kernel.o conv_rocks_cpp_default.o
	mv $@ $(BIN)
conv_rocks_cpp_direct: kernel.o conv_rocks_cpp_direct.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(ROCKS_SHARED_LIBS) $(ROCKS_EXTRA_FLAGS) -g --cudart static -link -o "$@" kernel.o conv_rocks_cpp_direct.o
	mv $@ $(BIN)


$(BIN):
	mkdir -p $(BIN)

.PHONY: clean

clean:
	rm -rf *.o ../bin/synthetic
