#include <nvfunctional>
#include <assert.h>

#include "gpukv/fs_calls.cu.h"
#include "gpukv/util.cu.h"

#include "bfs.cu.h"

__global__ void bfs_kernel(char *data, unsigned int size, int m,
		int *column_indices, DistT *dist, Worklist2 *queues, struct g_flag *flag){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long offset = (unsigned long)tid * size;
	int src;
	int n;
	int done = 0;
	int round = 0;
	
	do {
		while(tid > 0 && done && flag->nitems && flag->remain > 0 && flag->remain != flag->nitems);
		done = 0;
		if(flag->remain > 0 && queues[flag->idx].pop_id(tid, src)){
			int ret = key_value_op_thread((unsigned long)src, offset, size, GET);

			graph_parse_vertex(src, data, column_indices, offset, n);
			for(int i = 0; i < n; ++i){
				int dst = column_indices[tid * 128 + i];
				if((dist[dst] == MYINFINITY) && (atomicCAS(&dist[dst], MYINFINITY, dist[src]+1) == MYINFINITY)) {
					assert(queues[1-flag->idx].push(dst));
				}
			}
			done = 1;
			atomicSub(&flag->remain, 1);
		}
		if(tid == 0){
			while(flag->remain>0);
			done = 0;
			if(*(queues[1-flag->idx].d_index) == 0) printf("nitem is 0\n");
			flag->nitems = *(queues[1-flag->idx].d_index);
			//printf("round %d finish(next: %d)\n", round, flag->nitems);
			round++;
			flag->idx = 1 - flag->idx;
			*queues[1-flag->idx].d_index = 0;
			flag->remain = flag->nitems;
		}
	}while(flag->nitems > 0);
}

__global__ void bfs_no_gpukv(char *data, int* column_indices, int size,
		DistT *dist, Worklist2 in_queue, Worklist2 out_queue){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long offset = (unsigned long)tid * size;
	int num;
	int src;
	if (in_queue.pop_id(tid, src)) {
		graph_parse_vertex(src, data, column_indices, offset, num);
		for(int i = 0; i < num; ++i){
			int dst = column_indices[tid*128 + i];
			if ((dist[dst] == MYINFINITY) && (atomicCAS(&dist[dst], MYINFINITY, dist[src]+1) == MYINFINITY)) {
				assert(out_queue.push(dst));
			}
		}
	}
}

__global__ void insert(int source, Worklist2 queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) queue.push(source);
	return;
}


__global__ void inner_work(char *data, unsigned int size, int m, int *column_indices, DistT *dist, Worklist2 &in_queue, Worklist2 &out_queue){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long offset = (unsigned long)tid * size;
	int n;
	int src;
	if (in_queue.pop_id(tid, src)) {

		int ret = key_value_op_thread((unsigned long)src, offset, size, GET);
		graph_parse_vertex(src, data, column_indices, offset, n);
		for(int i = 0; i < n; ++i){
			int dst = column_indices[tid*128 + i];
			if ((dist[dst] == MYINFINITY) && (atomicCAS(&dist[dst], MYINFINITY, dist[src]+1) == MYINFINITY)) {
				assert(out_queue.push(dst));
			}
		}
	}

}

__global__ void inner_bfs(char *data, unsigned int size, int m, int *column_indices, DistT *dist, Worklist2 *queues){
	int nitems = 1;
	int idx = 0;
	int iter = 0;
	int nblocks, nthreads = BLOCK_SIZE;
	do{
		nblocks = (nitems - 1) / nthreads + 1;

		inner_work<<<nblocks, nthreads>>>(data, size, m, column_indices, dist, queues[idx], queues[1-idx]);
		cudaDeviceSynchronize();

		nitems = *(queues[1-idx].d_index);
		//printf("Round %d finished!(next: %d)\n", iter, nitems);
		idx = 1 - idx;
		*queues[1-idx].d_index = 0;
		iter++;
	}while(nitems > 0);
	
	
}

