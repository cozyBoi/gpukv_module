#ifndef _BFS_CU_H_
#define _BFS_CU_H_

#include "gardenia/worklistc.h"
#include "gardenia/common.h"
#include "gardenia/cuda_launch_config.hpp"
#include "gardenia/cutil_subset.h"


__global__ void bfs_kernel(char *data, unsigned int size, int m, int *column_indices, DistT *dist, Worklist2 *queues, struct g_flag *flag);

__global__ void bfs_no_gpukv(char *data, int* column_indices, int size, DistT *dist, Worklist2 in_queue, Worklist2 out_queue);

__global__ void insert(int source, Worklist2 queue);

__global__ void inner_work(char *data, unsigned int size, int m, int *column_indices, DistT *dist, Worklist2 &in_queue, Worklist2 &out_queue);

__global__ void inner_bfs(char *data, unsigned int size, int m, int *column_indices, DistT *dist, Worklist2 *queues);

#endif
