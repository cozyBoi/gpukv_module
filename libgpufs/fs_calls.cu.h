/* 
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */

/* 
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */

#ifndef FS_CALLS_CU_H
#define FS_CALLS_CU_H

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "timer.h"
#include "fs_globals.cu.h"


__device__ int key_value_op_thread(unsigned long key, unsigned long offset, unsigned int size, unsigned int mode);
__device__ int key_value_op_warp(unsigned long key, unsigned long offset, unsigned int size, unsigned int mode);
__device__ int key_value_op_block(unsigned long key, unsigned long offset, unsigned int size, unsigned int mode);

__device__ int key_value_op_thread_bad1(unsigned long key, unsigned long offset,unsigned int size, unsigned int mode);

__device__ int key_value_op_block_bad1(unsigned long key, unsigned long offset,	unsigned int size, unsigned int mode);
__device__ int key_value_op_block_bad2(unsigned long key, unsigned long offset,unsigned int size, unsigned int mode);
#endif

////////////////////////////////////////////////////////
