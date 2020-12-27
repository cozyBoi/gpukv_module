#include "gardenia/common.h"

#include "spmv.cu.h"

__global__ void spmv_kernel(char *data, unsigned int size, int tot_vert, int *column_indices,
		const ValueT * x, ValueT *y, ValueT *Ax, ValueT *Aj, struct g_flag *flag){
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = tot_vert / NUM_THREADS;
	int start_row = row * stride, end_row = row + stride - 1;

	int i, n;

	unsigned long offset = (unsigned long)tid * size;


	for(i = start_row; i <= end_row; ++i){
		if(i < tot_vert){
			ValueT sum = y[start_row];
			key_value_op_thread((unsigned long)i, offset, size, GET); 
			graph_parse_vertex_all(i, data, column_indices, /*weight*/, offset, n); 
			for(int j = 0; j < n; ++j){
				sum += Ax[tid*128+j] * x[Aj[tid*128+j]];
			}
			y[start_row] = sum;
		}
	}
}
