ROOTDIR?=../..
ROCKS_CONFIG:=$(ROOTDIR)/rocksdb/make_config.mk

include $(ROCKS_CONFIG)

INCLUDEDIR=$(ROOTDIR)/include
LIBDIR=$(ROOTDIR)/lib
#SHARED_LIBS=-lrocksdb -ldl -lpthread
SHARED_LIBS=-ldl -lpthread
ROCKS_SHARED_LIBS=$(SHARED_LIBS) -lz -lrt


CXXFLAGS=-L$(LIBDIR) -lrocksdbfake $(ROCKS_SHARED_LIBS) -I$(INCLUDEDIR) --std=c++11
CXXFLAGS2=-L$(LIBDIR) -lrocksdb $(ROCKS_SHARED_LIBS) -I$(INCLUDEDIR) --std=c++11
PLATFORM_CXXFLAGS= -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX  -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DZLIB -DNUMA -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DHAVE_SSE42  -DHAVE_PCLMUL  -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL

CUDA_ROOT:=/usr/local/cuda
CUDA_SHARED_LIBS=-lcudart -lcuda -ldl -lgpufs
CUDA_INCLUDE=-I$(CUDA_ROOT)/include -I$(INCLUDEDIR)
CUDA_LIB=-L$(LIBDIR) -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib -L /usr/lib64/nvidia -L /usr/lib/nvidia $(CUDA_SHARED_LIBS)
NVCC=$(CUDA_ROOT)/bin/nvcc
NVCCFLAGS=-g -G -O2 $(CUDA_LIB) $(CUDA_INCLUDE) $(EXTRA_FLAGS) --std=c++11 --generate-code code=sm_61,arch=compute_61 -maxrregcount 32





all: get_bd_modified get_bd_original get_p2p_time

get_bd_modified.o: get_bd.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS) -g -dc

get_bd_modified: get_bd_modified.o
	$(CXX) -o "$@" $< $(CXXFLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS)

get_bd_original.o: get_bd.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS2) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS) -g -dc

get_bd_original: get_bd_original.o
	$(CXX) -o "$@" $< $(CXXFLAGS2) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS)

get_p2p_time.o: get_p2p_time.cu
	$(NVCC) $(NVCCFLAGS) -g -dc -c $<

get_p2p_time: get_p2p_time.o
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" $<

clean:
	rm -rf *.o get_bd_modified get_bd_original get_p2p_time
