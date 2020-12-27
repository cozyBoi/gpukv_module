include $(ROCKS_CONFIG)
include ../common.mk

OBJS= spmv.o conv.o conv_rocks.o gpukv.o

all: spmv_gpukv spmv_conv spmv_conv_rocks

# Objective files
spmv.o: spmv.cu
	$(NVCC) $(NVCCFLAGS) -g -dc $< -o $@
conv.o: conv.cu
	$(NVCC) $(NVCCFLAGS)  -g -dc $< -o $@ 
conv_rocks.o: conv.cu
	$(NVCC) $(ROCKS_FLAGS) $(NVCCFLAGS) $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -g -dc $< -o $@ -lz -lrt
gpukv.o: gpukv.cu
	$(NVCC) $(NVCCFLAGS) -g -dc $< -o $@

# Executable files
spmv_gpukv: spmv.o gpukv.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" spmv.o gpukv.o
	mv $@ $(BIN) 
spmv_conv: spmv.o conv.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" spmv.o conv.o
	mv $@ $(BIN)
spmv_conv_rocks: spmv.o conv_rocks.o | $(BIN)
	$(NVCC) $(PLATFORM_LDFLAGS) $(PLAGFORM_CXXFLAGS) $(EXEC_LDFLAGS) $(NVCCFLAGS) -g --cudart static -link -o "$@" spmv.o conv_rocks.o -lz -lrt
	mv $@ $(BIN)

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -f *.o
