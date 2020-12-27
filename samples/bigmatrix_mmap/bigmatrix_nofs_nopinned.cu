/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/


#define FETCH_SIZE 4096
__shared__ float s_vector[FETCH_SIZE];
#define ACCUM_N 512
__shared__ volatile float s_reduction[ACCUM_N];

__device__ void inner_product_simple( float* a,  float* b, int size,float* res)
{
        float tmp=0;
        __syncthreads();
        if (threadIdx.x==0) {
                        *res=0;
        }
        __syncthreads();
        int i=0;
        for( i=threadIdx.x;i<size;i+=blockDim.x){

                tmp+=(a[i]*b[i]);
        }

        atomicAdd((float*)res,tmp);
        __syncthreads();

}




__device__ float inner_product( float* a,  float* b, int size)
{
	float tmp=0;
//	__syncthreads();
//	if (threadIdx.x==0) {
//			*res=0;
//	}
//	__syncthreads();
	int i=0;
	for( i=threadIdx.x;i<size;i+=blockDim.x){

		tmp+=a[i]*b[i];
	}
	s_reduction[threadIdx.x]=tmp;
	
         __syncthreads();
	for (int stride = ACCUM_N / 2; stride > 32; stride >>= 1)
        {
	    if(threadIdx.x<stride) s_reduction[threadIdx.x] += s_reduction[stride + threadIdx.x];
	    __syncthreads();
        }
	for (int stride = 32; stride > 0 && threadIdx.x<32 ; stride >>=1 )
	{
		if(threadIdx.x<stride) s_reduction[threadIdx.x] += s_reduction[stride + threadIdx.x];
	}

	__syncthreads();
	
	return s_reduction[0];
	
}
__device__ void prefetch(float* s_v, float* v,int size){
	for(int i=threadIdx.x;i<size;i+=blockDim.x)
	{
		s_v[i]=v[i];
	}
}

__shared__ float tmp_res_simple;
__global__ void bigmatrix_nofiles_simple(float* m, float *v, float* o,int out_offset, int m_size,int v_size)
{
        int per_block=m_size/gridDim.x;

                for(size_t out=per_block*blockIdx.x;out<(blockIdx.x+1)*per_block;out+=v_size){

                                inner_product_simple(v,m+out, v_size,&tmp_res_simple);
                                if(threadIdx.x==0)
                                {
                                        (*(o+out_offset + (out/v_size)))=tmp_res_simple;
                                }
                                __syncthreads();
                }
}

__global__ void bigmatrix_nofiles(float* m, float *v, float* o,int out_offset, int m_size, int v_size)
{
	int per_block=m_size/gridDim.x;
	
	//for (int i=0;i<10000;i++){
	for (int tile=0;tile<v_size;tile+=FETCH_SIZE){

		prefetch(s_vector,v+tile,FETCH_SIZE);
		__syncthreads();

		for(size_t out=per_block*blockIdx.x;out<(blockIdx.x+1)*per_block;out+=v_size){
		
			float tmp_res=inner_product(s_vector,m+out+tile, FETCH_SIZE);
			if(threadIdx.x==0)
			{
				if (tile==0) (*(o+out_offset + (out/v_size)))=tmp_res;
				else (*(o+out_offset + (out/v_size)))+=tmp_res;
			}
			__syncthreads();
		}
	}
	//}
}

void init_device_app(){

//	CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1<<30));
}

#include <sys/time.h>

      double _timestamp(){
                struct timeval tv;
                gettimeofday(&tv,0);
                return 1e6*tv.tv_sec+tv.tv_usec;
        }

#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "timer.h"
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <errno.h>
#include <string.h>

void*  open_map_file(const char* f, int* fd, size_t* size, int type, bool do_mmap=true)
{
        int open_fd=open(f,type==O_RDONLY?type:type|O_CREAT|O_TRUNC,S_IRUSR|S_IWUSR);



        if (open_fd<0){
                 perror("open failed");
                return NULL;
        }
        if (type!=O_RDONLY) {
                assert(*size>0);
                if (ftruncate(open_fd,*size)){
                        perror("ftrunc failed");
                        return NULL;
                }
        }

        struct stat s;
        if (fstat(open_fd,&s)) {
                        fprintf(stderr,"Problem with fstat the file on CPU: %s \n ",strerror(errno));
        }

        if (s.st_size==0) {
                fprintf(stderr,"file with zero lenght, skipping %s\n",f);
                close(open_fd);
                return NULL;
        }
        void* data=NULL;
	if (do_mmap){
		data=mmap(NULL,s.st_size,type==O_RDONLY?PROT_READ:PROT_READ|PROT_WRITE,MAP_POPULATE|(O_RDONLY?MAP_PRIVATE:MAP_SHARED),open_fd,0);
	        if (data==MAP_FAILED)   {
	                perror("mmap");
	                close(open_fd);
        	        return NULL;
	        }
	}
        
	*fd=open_fd;
        *size=s.st_size;

        return data;
}


#define CUDA_SAFE_CALL(x) if((x)!=cudaSuccess) { fprintf(stderr,"CUDA ERROR %s: %d %s\n",__FILE__, __LINE__, cudaGetErrorString(cudaGetLastError())); exit(-1); }

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "timer.h"
//DEBUG


#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>


void unmap_close_file(int fd, void* ptr,size_t len, const char* what)
{

        if(munmap(ptr,len)) {  perror("unmap"); fprintf(stderr,"%s\n",what); };
        close(fd);
}

#include <assert.h>

const int NUM_STREAMS=16;
// size of the output used for data staging
int main( int argc, char** argv)
{
	if(argc<4) {
		fprintf(stderr," <vector> <matrix> <output>\n\n");
		return -1;
	}
	
	double total_time=0;

	int nthreads=256;
int trials =5;
for(int t=-1;t<trials+1;t++){


	int fd_m;
	size_t size_m;
	
	char* h_matrix=(char*)open_map_file(argv[2], &fd_m, &size_m, O_RDONLY,false);
//	assert(h_matrix);




	float* h_d_matrix[NUM_STREAMS];
	float* d_matrix[NUM_STREAMS];
	
	int data_per_chunk=146800640/2;


	
	//size_t data_per_chunk=size_m/NUM_STREAMS;
	

	assert(size_m%data_per_chunk==0);



	printf("Data per chunk: %lu\n",data_per_chunk);
	assert(data_per_chunk%sizeof(float)==0);
	
	for (int i=0;i<NUM_STREAMS;i++){
		CUDA_SAFE_CALL(cudaHostAlloc(&h_d_matrix[i], data_per_chunk,  cudaHostAllocDefault));
        	CUDA_SAFE_CALL(cudaMalloc(&d_matrix[i],data_per_chunk));
	}
	
        double time_before=_timestamp();
	if (t<=0) time_before=0;	
	
	int fd_v;
	size_t size_v;

	char* h_vector=(char*)open_map_file(argv[1],&fd_v,&size_v,O_RDONLY);
	assert(h_vector);




//	float* h_d_vector;
	float* d_vector;
//        CUDA_SAFE_CALL(cudaHostAlloc(&h_d_vector,size_v,  cudaHostAllocDefault));
        CUDA_SAFE_CALL(cudaMalloc(&d_vector,size_v));
	
	int values_per_block=10;
	int nblocks=(data_per_chunk/size_v/values_per_block);

	assert(data_per_chunk/size_v/nblocks>0);
	assert((data_per_chunk/size_v)%nblocks==0);
	

	printf("Running with %d blocks, %d threads, %d vals per block\n",nblocks, nthreads,(data_per_chunk/size_v)/nblocks );

	int fd_v_out;
	size_t size_v_out=size_m/size_v*sizeof(float);
	assert(size_v_out);
	
	char* h_v_out=(char*)open_map_file(argv[3], &fd_v_out, &size_v_out, O_RDWR);
	assert(h_v_out);
	float* h_d_v_out;
	float* d_v_out;
        CUDA_SAFE_CALL(cudaHostAlloc(&h_d_v_out,size_v_out, cudaHostAllocDefault));
        CUDA_SAFE_CALL(cudaMalloc(&d_v_out,size_v_out));
	
	fprintf(stderr,"using: %s for matrix of size %lu, %s for vector of size %lu, %s for output of size %lu, data per chunk %lu\n",
				argv[2], size_m,argv[1],size_v,argv[3],size_v_out,data_per_chunk);
	
	
       	cudaStream_t s[NUM_STREAMS];
	for(int i=0;i<NUM_STREAMS;i++){
	        CUDA_SAFE_CALL(cudaStreamCreate(&s[i]));
	}
        
        
	int c=0;

//	memcpy(h_d_vector,h_vector,size_v);
	CUDA_SAFE_CALL(cudaMemcpy(d_vector,h_vector,size_v,cudaMemcpyHostToDevice));
	
	for(size_t i=0 ;i<size_m;i+=data_per_chunk)
	{
		fprintf(stderr,"chunk %lu %d\n",i, i/data_per_chunk);
	
		size_t total_read=0;
		CUDA_SAFE_CALL(cudaStreamSynchronize(s[c]));
		total_read=pread(fd_m, (char*)(h_d_matrix[c]), data_per_chunk, i);

	//	CUDA_SAFE_CALL(cudaMemcpyAsync(((char*)d_matrix[i/data_per_chunk]),h_d_matrix[i/data_per_chunk],
	//			data_per_chunk,cudaMemcpyHostToDevice,s[c]));
	/*	for (size_t z=0;z<data_per_chunk;z+=(1<<20)){
			total_read=pread(fd_m, (char*)(h_d_matrix[i/data_per_chunk])+z, 1<<20,  i+z);
			if (total_read!=(1<<20)){
				perror ("failed to read");
			}
			CUDA_SAFE_CALL(cudaMemcpyAsync(((char*)d_matrix[i/data_per_chunk])+z,(char*)(h_d_matrix[i/data_per_chunk])+z,1<<20,cudaMemcpyHostToDevice,s[c]));
		//	CUDA_SAFE_CALL(cudaMemcpy((char*)(d_matrix[i/data_per_chunk])+z,h_d_matrix[i/data_per_chunk],1<<20,cudaMemcpyHostToDevice)); 
		}
	*/
		CUDA_SAFE_CALL(cudaMemcpyAsync((char*)(d_matrix[c]),h_d_matrix[c],
						data_per_chunk,cudaMemcpyHostToDevice,s[c])); 
	
		bigmatrix_nofiles_simple<<<nblocks,nthreads,0,s[c]>>>(d_matrix[c],d_vector,d_v_out,i/size_v,
								data_per_chunk/(sizeof(float)), size_v/(sizeof(float)));
	//	bigmatrix_nofiles_simple<<<nblocks,nthreads,0,0>>>(d_matrix[i/data_per_chunk],d_vector,d_v_out,i/size_v,
	//							data_per_chunk/(sizeof(float)), size_v/(sizeof(float)));
		
		CUDA_SAFE_CALL(cudaMemcpyAsync(h_d_v_out+i/size_v,d_v_out+i/size_v,data_per_chunk/size_v*sizeof(float),cudaMemcpyDeviceToHost,s[c]));
		c++;
		c%=NUM_STREAMS;
	}				
	//CUDA_SAFE_CALL(cudaMemcpy(h_v_out,d_v_out,size_v_out,cudaMemcpyDeviceToHost));


       cudaError_t error = cudaDeviceSynchronize();

 	   //Check for errors and failed asserts in asynchronous kernel launch.
	    if(error != cudaSuccess )
	    {
	        printf("Device failed, CUDA error message is: %s\n\n", cudaGetErrorString(error));
	    }
	
	
	memcpy(h_v_out,h_d_v_out,size_v_out);

	cudaFree(d_v_out);
	cudaFree(d_vector);
	cudaFree(d_matrix);
	unmap_close_file(fd_v_out,h_v_out,size_v_out,"out");

	//unmap_close_file(fd_m,h_matrix,size_m,"matrix");
	close(fd_m);
	unmap_close_file(fd_v,h_vector,size_v,"vector");
	
	double time_after=_timestamp();
	total_time+=(time_after-time_before);
	if(t<=0) total_time=0;
	if (t>0) fprintf(stderr,"total time %.0f us, avg %.0f us, bw %.3f GB/s \n ", time_after-time_before, total_time/t, t*1.0e6*(size_v+size_m+size_v_out)/total_time/(1024.0*1024.0*1024.0));

 //       cudaFreeHost(h_d_v_out);
 //       cudaFreeHost(h_d_vector);
  //      cudaFreeHost(h_d_matrix);

	//	cudaFree(d_output);	
	cudaDeviceReset();
//	if(error) break;
}
	return 0;
}


