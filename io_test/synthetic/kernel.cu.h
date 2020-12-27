#ifndef __KERNEL_CU_H__

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void init_curand_state(curandState_t* states);
//__global__ void streaming_gpukv(char *data, unsigned int req_size, int tot_req, int stride, int iter, double portion, int *alive, int *alive2);
__global__ void streaming_gpukv(char *data, unsigned int req_size, int tot_req, int stride, int iter, double portion);

__global__ void dynamic_gpukv(char *data, unsigned int req_size, int *req_list, int tot_req, int stride, int iter, double portion, int blocks, int threads, curandState_t* states);

__global__ void inner_dynamic(char *data, unsigned int req_size, int *req_list, int req_num, int tot_req, int iter, double portion, curandState_t* states);

__global__ void streaming_conv(char *data, unsigned int req_size, int stride, int iter, double portion);

__global__ void dynamic_conv(char *data, unsigned int req_size, int *req_list, int req_num, int tot_req, int iter, double portion, curandState_t* states);


#endif
