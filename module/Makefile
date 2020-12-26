obj-m = key_module.o
#KVROOT_DIR=/home/asknor/kv_gpufs
#NVIDIA_SRC_DIR=/usr/src/nvidia-418.56/nvidia
NVIDIA_SRC_DIR=/usr/src/linux-headers-4.9.140-tegra-ubuntu18.04_aarch64/nvidia
ccflags-y += -I$(NVIDIA_SRC_DIR) -I$(PWD) #-I$(KVROOT_DIR)/include
#KDIR := /usr/src/linux-4.1.52/ 
KDIR := /usr/src/linux-headers-4.9.140-tegra-ubuntu18.04_aarch64/kernel-4.9/
PWD := $(shell pwd)
#KBUILD_EXTRA_SYMBOLS := /usr/src/nvidia-418.56/Module.symvers
KBUILD_EXTRA_SYMBOLS := /usr/src/linux-headers-4.9.140-tegra-ubuntu18.04_aarch64/kernel-4.9/Module.symvers
ifeq ($(t),1)   #block
	EXTRA_CFLAGS=-DRW_IPC_SIZE=256
else
ifeq ($(t),3) #thread
	EXTRA_CFLAGS=-DRW_IPC_SIZE=28672
else
ifeq ($(t),2)
	EXTRA_CFLAGS=-DRW_IPC_SIZE=1024
else
	EXTRA_CFLAGS=-DRW_IPC_SIZE=$(d)
endif
endif
endif

ifneq ($(l),0) #timing
	EXTRA_CFLAGS+=-DMALLOC_STATS -DTIMING_STATS
endif

$(info VAR="$(EXTRA_CFLAGS)")

INSTALLDIR?=~/atc/kv_gpufs
key_module :
	$(MAKE) -C $(KDIR) $(MAKE_PARAMS) M=$(PWD) modules

install:
	install -m 0644 *.h $(INSTALLDIR)/include

clean :
	rm -rf *.o
	rm -rf *.ko
	rm -rf *.mod.c
	rm -rf *.order
	rm -rf *.symvers

