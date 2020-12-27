#!/bin/bash
#cp /usr/src/linux-4.17.6/arch/x86/entry/syscalls/syscall_32.tbl ./
#cp /usr/src/linux-4.17.6/include/linux/compat.h ./


cp syscall_64.tbl /usr/src/linux-4.1.52/arch/x86/syscalls/syscall_64.tbl
cp syscalls.h /usr/src/linux-4.1.52/include/linux/syscalls.h
cp key_value.c /usr/src/linux-4.1.52/kernel/key_value.c 
cp Makefile /usr/src/linux-4.1.52/kernel/Makefile
cp nvme.h /usr/src/linux-4.1.52/include/linux/nvme.h 
cp nvme-core.c /usr/src/linux-4.1.52/drivers/block/nvme-core.c 
