#include <stdio.h>

#include "gpukv/fs_constants.h"
#include "gpukv/fs_calls.cu.h"
#include "gpukv/util.cu.h"
#include "gpukv/fs_debug.cu.h"

#include <curand.h>
#include <curand_kernel.h>

#include "kernel.cu.h"

__global__ void init_curand_state(curandState_t* states){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(17237, tid, 0, &states[tid]);
}


__global__ void streaming_gpukv(char *data, unsigned int req_size,
        //int tot_req, int stride, int iter, double portion, int *alive, int *alive2){
        int tot_req, int stride, int iter, double portion){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long offset = (unsigned long)tid * req_size;
    int calc = req_size * portion;
    int round = tot_req / stride;

    //for(int i = tid; i < tot_req; i+=stride){
    //atomicAdd(alive, 1);
    for(int i = 0; i < round; i++){
        key_value_op_thread((unsigned long)(i*stride+tid), offset, req_size, GET);
        //if(stride == 0) key_value_op_thread((unsigned long)(stride+tid), offset, req_size, GET);
        for(int j = 0; j < calc; ++j){
            for(int k = 0; k < iter; ++k){
                data[offset+j] = data[offset+j] * data[offset+j];
            }
        }
    }
    //atomicAdd(alive2, 1);
}

__global__ void dynamic_gpukv(char *data, unsigned int req_size, int *req_list,
        int tot_req, int stride, int iter, double portion, int blocks, int threads, curandState_t *states){
    int round = tot_req / stride;
    for(int i = 0; i < round; ++i){
        inner_dynamic<<<blocks, threads>>>(data, req_size, req_list, stride, tot_req, iter, portion, states);
        cudaDeviceSynchronize();
    }
}

__global__ void inner_dynamic(char *data, unsigned int req_size, int *req_list, 
        int req_num, int tot_req, int iter, double portion, curandState_t *states){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t tmp = 0;
    unsigned long offset = (unsigned long)tid * req_size;
    int calc = req_size *portion;

    key_value_op_thread((unsigned long)req_list[tid], offset, req_size, GET);
    for(int j = 0; j < calc; ++j){
        for(int k  = 0; k < iter; ++k){
            data[offset+j] = data[offset+j] * data[offset+j];
        }
        tmp += data[offset+j];
    }
    //unsigned int next = curand(&states[tid]) % tot_req;
    //req_list[tid] = (tmp * tmp) % tot_req;
    req_list[tid] = curand(&states[tid]) % tot_req;
}

__global__ void streaming_conv(char *data, unsigned int req_size, int stride, int iter, double portion){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long offset = (unsigned long)tid * req_size;
    int calc = req_size * portion;
    for(int j = 0; j < calc; ++j){
        for(int k = 0; k < iter; ++k){
            data[offset+j] = data[offset+j] * data[offset+j];
        }
    }
}

__global__ void dynamic_conv(char *data, unsigned int req_size, int *req_list, 
        int req_num, int tot_req, int iter, double portion, curandState_t *states){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t tmp = 0;
    unsigned long offset = (unsigned long)tid * req_size;
    int calc = req_size * portion;

    for(int j = 0; j < calc; ++j){
        for(int k  = 0; k < iter; ++k){
            data[offset+j] = data[offset+j] * data[offset+j];
        }
        tmp += data[offset+j];
    }
    //req_list[tid] = (tmp * tmp) % (uint64_t)tot_req;
    req_list[tid] = curand(&states[tid]) % tot_req;
}
