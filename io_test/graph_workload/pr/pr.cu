#include <cub/cub.cuh>

#include "gpukv/fs_calls.cu.h"
#include "gpukv/util.cu.h"

#include "gardenia/common.h"

#include "pr.cu.h"
typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;

// init score : 1 / tot_vert
__global__ void pr_kernel(char *data, unsigned int size, int tot_vert, int *column_indices, int *degree, 
		ScoreT *outgoing_contrib, ScoreT *scores, ScoreT base_score, struct g_flag *flag){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float block_sum;
	float error = 0;
	int stride = tot_vert / NUM_THREADS;
	int init_vertex = tid * stride;
	//int end_vertex = ((init_vertex + stride - 1) > tot_vert) ? tot_vert : (init_vertex + stride - 1);
	int end_vertex = init_vertex + stride - 1;
	int vertex;
	int n;
	unsigned long offset = (unsigned long)tid * size;
	for(vertex = init_vertex; vertex <= end_vertex; ++vertex){
		if(vertex < tot_vert){
			key_value_op_thread((unsigned long)vertex, offset, size, GET);
			graph_parse_vertex(vertex, data, column_indices, offset, n);
			degree[vertex] = n;
		}
	}
	__syncthreads();

	do{
		if(tid == 0) {
			flag->diff = 0;
			printf("[DEBUG] !\n");
		}
		while(tid > 0 && flag->diff != 0);
		for(vertex = init_vertex; vertex <= end_vertex; ++vertex){
			if(vertex < tot_vert){
				outgoing_contrib[vertex] = (degree[vertex] == 0) ? 0.0f : (scores[vertex] / degree[vertex]);
			}
		}
		__syncthreads();

		for(vertex = init_vertex; vertex <= end_vertex; ++vertex){
			error = 0;
			if(vertex < tot_vert){
				key_value_op_thread((unsigned long)vertex, offset, size, GET);
				graph_parse_vertex(vertex, data, column_indices, offset, n);

				ScoreT incoming_total = 0;
				for(int i = 0; i < n; ++i){
					int dst = column_indices[tid*128+i];
					incoming_total += outgoing_contrib[dst];
				}
				ScoreT old_score = scores[vertex];
				scores[vertex] = base_score + kDamp * incoming_total; 
				error = fabs(scores[vertex] - old_score);
			}
			__syncthreads();

			block_sum = BlockReduce(temp_storage).Sum(error);

			if(threadIdx.x == 0){
				atomicAdd(&flag->diff, block_sum);
			}
			vertex++; 
		}
		__syncthreads();
		if(tid == 0){
			flag->round++;
			printf(" %2d    %f\n", flag->round, flag->diff);
		}
	}while(flag->diff > EPSILON && flag->round < MAX_ITER);
}


/********* conventional page rank functions *********/

__global__ void set_degree(char *data, int m, int *column_indices, int stride, int *degree, int size){
	int tid	= blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid + stride;
	unsigned long offset = (unsigned long)tid * size;
	int num;
	if(src < m){
		graph_parse_vertex(src, data, column_indices, offset, num);
		degree[src] = num;
	}
}

__global__ void contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < m) outgoing_contrib[u] = degree[u] == 0 ? 0 : scores[u] / degree[u];
}

__global__ void pull_fused(char *data, int m, int size, int stride, IndexT *column_indices, 
		ScoreT *scores, ScoreT *outgoing_contrib, float *diff, ScoreT base_score) {
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid + stride;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float error = 0;
	int num;
	unsigned long offset = (unsigned long)tid * size;
	if (src < m) {
		graph_parse_vertex(src, data, column_indices, offset, num);
		ScoreT incoming_total = 0;
		for(int i = 0; i < num; ++i){
			IndexT dst = column_indices[tid*128 + i];
			incoming_total += outgoing_contrib[dst];
		}
		ScoreT old_score = scores[src];
		scores[src] = base_score + kDamp * incoming_total;
		error += fabs(scores[src] - old_score);
	}
	float block_sum = BlockReduce(temp_storage).Sum(error);
	if(threadIdx.x == 0) {
		atomicAdd(diff, block_sum);
	}
}


__global__ void inner_set_degree(char *data, int tot_vert, int *column_indices, int stride, int *degree, int size){
	int tid	= blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid + stride;
	unsigned long offset = (unsigned long)tid * size;
	int num;
	if(src < tot_vert){
		key_value_op_thread((unsigned long)src, offset, size, GET);
		graph_parse_vertex(src, data, column_indices, offset, num);
		degree[src] = num;
	}
}


__global__ void inner_pull_fused(char *data, int tot_vert, unsigned int size, int stride,
		int *column_indices, ScoreT *outgoing_contrib, ScoreT *scores, float *diff, ScoreT base_score){
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid + stride;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float error = 0;
	int num;
	unsigned long offset = (unsigned long)tid * size;
	if (src < tot_vert) {
		key_value_op_thread((unsigned long)src, offset, size, GET);
		graph_parse_vertex(src, data, column_indices, offset, num);
		ScoreT incoming_total = 0;
		for(int i = 0; i < num; ++i){
			IndexT dst = column_indices[tid*128 + i];
			incoming_total += outgoing_contrib[dst];
		}
		ScoreT old_score = scores[src];
		scores[src] = base_score + kDamp * incoming_total;
		error += fabs(scores[src] - old_score);
	}
	float block_sum = BlockReduce(temp_storage).Sum(error);
	if(threadIdx.x == 0) {
		atomicAdd(diff, block_sum);
	}
}



__global__ void inner_kernel(char *data, unsigned int size, int tot_vert, int max_req, int *column_indices,
		int*degree, ScoreT *outgoing_contrib, ScoreT *scores, ScoreT base_score, float *diff){
	int stride;
	int nblocks, nthreads = BLOCK_SIZE;
	int iter = 0;

	nblocks = max_req / nthreads;
	for(stride = 0; stride < tot_vert; stride += max_req){
		inner_set_degree<<<nblocks, nthreads>>>(data, tot_vert, column_indices, stride, degree, size);
		cudaDeviceSynchronize();
	}
	//printf("[DEBUG] set degree\n");

	do{
		iter++;
		*diff = 0;
		nblocks = (tot_vert - 1) / nthreads + 1;
		contrib<<<nblocks, nthreads>>>(tot_vert, scores, degree, outgoing_contrib);
		cudaDeviceSynchronize();

		nblocks = max_req / nthreads;
		for(stride = 0; stride < tot_vert; stride += max_req){
			inner_pull_fused<<<nblocks, nthreads>>>(data, tot_vert, size, stride, column_indices, 
					outgoing_contrib, scores, diff, base_score);
			cudaDeviceSynchronize();
		}
		//printf(" %2d   %f\n", iter, *diff);

	} while(*diff > EPSILON && iter < MAX_ITER);
    printf("iter: %d\n", iter);
}
