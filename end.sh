#!/bin/bash


#build
rmmod key_module.ko
cd ./libgpufs && make clean
cd -
cd ./module && make clean
cd -
cd ./io_test && make clean
cd -



