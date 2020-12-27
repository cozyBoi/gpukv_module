#/*
#* This expermental software is provided AS IS.
#* Feel free to use/modify/distribute,
#* If used, please retain this disclaimer and cite
#* "GPUfs: Integrating a file system with GPUs",
#* M Silberstein,B Ford,I Keidar,E Witchel
#* ASPLOS13, March 2013, Houston,USA
#*/


# This Makefile is used to build the library:
# Run: make libgpufs.a
# For debug version: make dbg=1 libgpufsd.a
# This file is also included in the makefiles to build user applications
# See "workloads" directory
LIBGPUFSDIR?=.
INSTALLDIR?=..
REG?=32

CUDA_ROOT:=/usr/local/cuda
CUDA_INCLUDE=-I$(CUDA_ROOT)/include -I$(LIBGPUFSDIR)

ifneq ($(dbg),1)
	MACROS=-DDEBUG_NOINLINE="" 
	NVCC=$(CUDA_ROOT)/bin/nvcc -std=c++11 --generate-code code=sm_61,arch=compute_61 $(CUDA_INCLUDE) $(MACROS) -O3 -maxrregcount $(REG)
	OUTPUTDIR=release
else
	MACROS=-DDEBUG_NOINLINE="__noinline__" -DMALLOC_STATS -DTIMING_STATS -DDEBUG -g
	NVCC=$(CUDA_ROOT)/bin/nvcc -std=c++11 --generate-code code=sm_61,arch=compute_61 $(CUDA_INCLUDE) $(MACROS) -O0 -maxrregcount $(REG) --device-debug --debug -G
	OUTPUTDIR=debug
endif

ifeq ($(t),1)   #block
	EXTRA_FLAGS=-DRW_IPC_SIZE=256
else
ifeq ($(t),3) #thread
	EXTRA_FLAGS=-DRW_IPC_SIZE=28672
else
ifeq ($(t),2)
	EXTRA_FLAGS=-DRW_IPC_SIZE=1024
else
	EXTRA_FLAGS=-DRW_IPC_SIZE=$d
endif
endif
endif

ifneq ($(l),0) #timing
	EXTRA_FLAGS+=-DMALLOC_STATS -DTIMING_STATS
endif


CC=g++ $(CUDA_INCLUDE) -Wall -Wno-write-strings $(MACROS) $(EXTRA_FLAGS)
NVCC+= $(EXTRA_FLAGS)

GPUFS_SOURCES=\
$(OUTPUTDIR)/cpu_ipc.o\
$(OUTPUTDIR)/fs_calls.o\
$(OUTPUTDIR)/fs_initializer.o\
$(OUTPUTDIR)/timer.o\
$(OUTPUTDIR)/fs_debug.o\


ifneq ($(dbg),1)
$(OUTPUTDIR)/libgpufs.a : $(GPUFS_SOURCES) | $(OUTPUTDIR)
	ar ru $(OUTPUTDIR)/libgpufs.a $(GPUFS_SOURCES)
else
$(OUTPUTDIR)/libgpufsd.a: $(GPUFS_SOURCES) | $(OUTPUTDIR)
	ar ru $(OUTPUTDIR)/libgpufsd.a $(GPUFS_SOURCES)
endif

$(OUTPUTDIR)/timer.o: timer.cpp timer.h | $(OUTPUTDIR)
	$(CC) timer.cpp -c -o $(OUTPUTDIR)/timer.o

$(OUTPUTDIR)/cpu_ipc.o: cpu_ipc.cu fs_constants.h \
	util.cu.h  fs_debug.cu.h cpu_ipc.cu.h | $(OUTPUTDIR)
	$(NVCC) -dc cpu_ipc.cu -o $(OUTPUTDIR)/cpu_ipc.o

$(OUTPUTDIR)/fs_calls.o: fs_calls.cu fs_debug.cu.h fs_constants.h util.cu.h  cpu_ipc.cu.h  \
	timer.h | $(OUTPUTDIR)
	$(NVCC) -dc fs_calls.cu -o $(OUTPUTDIR)/fs_calls.o

$(OUTPUTDIR)/fs_initializer.o: fs_initializer.cu fs_debug.cu.h fs_constants.h util.cu.h cpu_ipc.cu.h  \
	timer.h fs_initializer.cu.h | $(OUTPUTDIR)
	$(NVCC) -dc fs_initializer.cu -o $(OUTPUTDIR)/fs_initializer.o

$(OUTPUTDIR)/fs_debug.o: fs_debug.cu fs_debug.cu.h | $(OUTPUTDIR)
	$(NVCC) -dc fs_debug.cu -o $(OUTPUTDIR)/fs_debug.o

$(OUTPUTDIR):
	mkdir -p $(OUTPUTDIR)

ifneq ($(dbg),1)
install: $(OUTPUTDIR)/libgpufs.a
	mkdir -p $(INSTALLDIR)/lib
	mkdir -p $(INSTALLDIR)/include
	mkdir -p $(INSTALLDIR)/include/gpukv
	install -m 0755 $(OUTPUTDIR)/libgpufs.a $(INSTALLDIR)/lib
	install -m 0644 *.h $(INSTALLDIR)/include/gpukv
#	install -m 0644 *.h $(GPUFSLIB)/*.h $(INSTALLDIR)/include
else
install: $(OUTPUTDIR)/libgpufsd.a
	mkdir -p $(INSTALLDIR)/lib
	mkdir -p $(INSTALLDIR)/include
	mkdir -p $(INSTALLDIR)/include/gpukv
	install -m 0755 $(OUTPUTDIR)/libgpufsd.a $(INSTALLDIR)/lib
	install -m 0644 *.h $(INSTALLDIR)/include/gpukv
#	install -m 0644 *.h $(GPUFSLIB)/*.h $(INSTALLDIR)/include
endif

PHONY:  clean

clean:
	rm -rf $(OUTPUTDIR) *.o *.a


