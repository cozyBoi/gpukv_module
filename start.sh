#!/bin/bash

ulimit -Hn 1048576
ulimit -Sn 1048576

./io_test/set_cpu 4 3.5GHz

if [ "$#" -eq "0" ]
then
	t=1 #ipc_size flags 1:256, 2:1024, 3:28672, 4 : d
	l=0 #1: timing,requst flags,print 0:NULL
	d=16
else
	t=$1
	l=$2
	d=$3
fi

# For debug mode gpufs build,
#cd ./libgpufs && make t=$t l=$l d=$d dbg=1 && make install dbg=1

# For common mode gpufs build,
cd ./libgpufs && make t=$t l=$l d=$d && make install

cd -
cd ./module && make t=$t l=$l d=$d  && make install && insmod key_module.ko
cd -

sed -i '1710s/.*/INSTALL_PATH ?= ..\//' rocksdb/Makefile

cd ./rocksdb && make install
cd -
cd ./io_test && make t=$t l=$l d=$d
cd -

# To add cpu load,
#stress-ng --cpu 1 -l 95&
