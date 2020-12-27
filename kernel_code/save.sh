#!/bin/bash
#cp /usr/src/linux-4.17.6/arch/x86/entry/syscalls/syscall_32.tbl ./
cp /usr/src/linux-4.1.52/arch/x86/syscalls/syscall_64.tbl ./

cp /usr/src/linux-4.1.52/include/linux/syscalls.h ./
#cp /usr/src/linux-4.17.6/include/linux/compat.h ./
cp /usr/src/linux-4.1.52/include/linux/nvme.h ./
cp /usr/src/linux-4.1.52/kernel/key_value.c ./
cp /usr/src/linux-4.1.52/drivers/block/nvme-core.c ./
cp /usr/src/linux-4.1.52/kernel/Makefile  ./
