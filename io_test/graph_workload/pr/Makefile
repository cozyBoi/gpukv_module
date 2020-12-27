include $(ROCKS_CONFIG)
include ../common.mk

OBJS= pr.o conv.o conv_rocks.o gpukv.o

all: pr_gpukv pr_conv pr_conv_rocks pr_conv_rocks_direct

# Objective files
pr.o: pr.cu
	$(NVCC) $(NVCCFLAGS) -g -dc $< -o $@
conv.o: conv_pr.cu
	$(NVCC) $(NVCCFLAGS)  -g -dc $< -o $@ 
conv_rocks.o: conv_pr.cu
	$(NVCC) $(ROCKS_FLAGS) $(NVCCFLAGS) $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -g -dc $< -o $@ -lz -lrt
conv_rocks_direct.o: conv_pr.cu
	$(NVCC) $(ROCKS_FLAGS) -D DIRECTIO $(NVCCFLAGS) $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -g -dc $< -o $@ -lz -lrt
gpukv.o: gpukv_pr.cu
	$(NVCC) $(NVCCFLAGS) -g -dc $< -o $@

# Executable files
pr_gpukv: pr.o gpukv.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" pr.o gpukv.o
	mv $@ $(BIN) 
pr_conv: pr.o conv.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" pr.o conv.o
	mv $@ $(BIN)
pr_conv_rocks: pr.o conv_rocks.o | $(BIN)
	$(NVCC) $(PLATFORM_LDFLAGS) $(PLAGFORM_CXXFLAGS) $(EXEC_LDFLAGS) $(NVCCFLAGS) -g --cudart static -link -o "$@" pr.o conv_rocks.o -lz -lrt
	mv $@ $(BIN)

pr_conv_rocks_direct: pr.o conv_rocks_direct.o | $(BIN)
	$(NVCC) $(PLATFORM_LDFLAGS) $(PLAGFORM_CXXFLAGS) $(EXEC_LDFLAGS) $(NVCCFLAGS) -g --cudart static -link -o "$@" pr.o conv_rocks_direct.o -lz -lrt
	mv $@ $(BIN)

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -f *.o
