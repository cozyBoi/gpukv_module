include $(ROCKS_CONFIG)
include ../common.mk

all: bfs_gpukv bfs_conv bfs_conv_rocks bfs_conv_rocks_direct bfs_conv_rocks_cpp bfs_conv_rocks_cpp_direct

bfs.o: bfs.cu
	$(NVCC) $(NVCCFLAGS) -g -dc $< -o $@
gpukv.o: gpukv_bfs.cu
	$(NVCC) $(NVCCFLAGS) -g -dc $< -o $@
conv.o: conventional.cu
	$(NVCC) $(NVCCFLAGS) -g -dc $< -o $@
conv_rocks.o: conventional.cu
	$(NVCC) $(ROCKS_FLAGS) $(NVCCFLAGS) $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -g -dc $< -o $@ -lz -lrt
conv_rocks_direct.o: conventional.cu
	$(NVCC) $(ROCKS_FLAGS) -D DIRECTIO $(NVCCFLAGS) $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -g -dc $< -o $@ -lz -lrt
conv_rocks_cpp_direct.o: conventional.cu
	$(NVCC) $(ROCKS_FLAGS) -D DIRECTIO -D ROCKS_CXX $(NVCCFLAGS) $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -g -dc $< -o $@ -lz -lrt

conv_rocks_cpp.o: conventional.cu
	$(NVCC) $(ROCKS_FLAGS) -D ROCKS_CXX $(NVCCFLAGS) $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -g -dc $< -o $@ -lz -lrt



bfs_gpukv: bfs.o gpukv.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" gpukv.o bfs.o 
	mv $@ $(BIN)
bfs_conv: bfs.o conv.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) -g --cudart static -link -o "$@" conv.o bfs.o
	mv $@ $(BIN)
bfs_conv_rocks: bfs.o conv_rocks.o | $(BIN)
	$(NVCC) $(PLATFORM_LDFLAGS) $(PLAGFORM_CXXFLAGS) $(EXEC_LDFLAGS) $(NVCCFLAGS) -g --cudart static -link -o "$@" bfs.o conv_rocks.o -lz -lrt
	mv $@ $(BIN)
bfs_conv_rocks_direct: bfs.o conv_rocks_direct.o | $(BIN)
	$(NVCC) $(PLATFORM_LDFLAGS) $(PLAGFORM_CXXFLAGS) $(EXEC_LDFLAGS) $(NVCCFLAGS) -g --cudart static -link -o "$@" bfs.o conv_rocks_direct.o -lz -lrt
	mv $@ $(BIN)

bfs_conv_rocks_cpp: bfs.o conv_rocks_cpp.o | $(BIN)
	$(NVCC) $(PLATFORM_LDFLAGS) $(PLAGFORM_CXXFLAGS) $(EXEC_LDFLAGS) $(NVCCFLAGS) -g --cudart static -link -o "$@" bfs.o conv_rocks_cpp.o -lz -lrt
	mv $@ $(BIN)

bfs_conv_rocks_cpp_direct: bfs.o conv_rocks_cpp_direct.o | $(BIN)
	$(NVCC) $(PLATFORM_LDFLAGS) $(PLAGFORM_CXXFLAGS) $(EXEC_LDFLAGS) $(NVCCFLAGS) -g --cudart static -link -o "$@" bfs.o conv_rocks_cpp_direct.o -lz -lrt
	mv $@ $(BIN)



$(BIN):
	mkdir -p $(BIN)

clean:
	rm -f *.o


