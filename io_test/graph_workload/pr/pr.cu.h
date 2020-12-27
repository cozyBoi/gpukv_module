#ifndef _PR_CU_H_

#define _PR_CU_H_

#include "gardenia/common.h"

const float kDamp = 0.85;

#define EPSILON 0.0001
#define MAX_ITER 5


__global__ void pr_kernel(char *data, unsigned int size, int tot_vert, int *column_indices, int *degree, ScoreT *outgoing_contrib, ScoreT *scores, ScoreT base_score, struct g_flag *flag);

__global__ void set_degree(char *data, int m, int *column_indices, int stride, int *degree, int size);

__global__ void contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib);
__global__ void pull_fused(char *data, int m, int size, int stride, IndexT *column_indices, 
		ScoreT *scores, ScoreT *outgoing_contrib, float *diff, ScoreT base_score);

__global__ void inner_set_degree(char *data, int m, int *column_indicese, int stride, int *degree, int size);

__global__ void inner_pull_fused(char *data, int tot_vert, unsigned int size, int stride,
		int *column_indices, ScoreT *outgoing_contrib, ScoreT *scores, float *diff, ScoreT base_score);

__global__ void inner_kernel(char *data, unsigned int size, int tot_vert, int max_req, int *column_indices,
		int*degree, ScoreT *outgoing_contrib, ScoreT *scores, ScoreT base_score, float *diff);


#endif
