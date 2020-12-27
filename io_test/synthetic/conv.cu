#include <stdio.h>
#include <stdlib.h>

#include <random>

#include <algorithm>

#include <omp.h>

#include "gpukv/fs_calls.cu.h"
#include "gpukv/host_loop.h"
#include "gpukv/util.cu.h"

#include "key-p2p.h"
#include "key_api.h"

#ifdef DATA_ON_ROCKS
#include "rocksdb/c.h"

// C++
#include <fstream>
#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"
#include "rocksdb/iostats_context.h"
#include "rocksdb/perf_context.h"
#include "rocksdb/table.h"
#include "rocksdb/utilities/options_util.h"
#include "rocksdb/statistics.h"

using namespace rocksdb;
#endif

#include <curand.h>

#include "kernel.cu.h"

const char DBPath[] = "/home/asknor/nvme/data_synthetic";
const char *(mode_name[]) = {"streaming","dynamic"};

int thread = 0;
double get_time_val = 0;
unsigned long long clock_sum;
unsigned long long clock_par_req[256];
unsigned long long clock_par_min[256];
unsigned long long clock_par_max[256];

static __inline__ unsigned long long rdtsc(void){
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc": "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo | ( ((unsigned long long)hi) << 32) );
}



#ifdef DATA_ON_ROCKS
void get_data_rocks(char *user_get, char *cuda_memory, int from, int to,
        int memory_size, int request_size, DB *db){
    int num_thread = thread;
    ReadOptions ro = ReadOptions();
    if(thread == 0) num_thread = 512;

    struct timespec st, et;
    unsigned long long clock_start, clock_end;
    clock_start = rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &st);
    #pragma omp parallel for num_threads(num_thread) schedule(dynamic)
    for(int i = from; i < to; ++i){
        unsigned long long req_start, req_end;
        char key[100];
        unsigned long long tmp_clock;
        std::string value;
        sprintf(key, "%d", i);
        //db->Get(ReadOptions(), key, &value);
        req_start = rdtsc();
        db->Get(ro, key, &value);
        memcpy(user_get+(i-from)*request_size, value.data(), value.size());
        req_end = rdtsc();
        tmp_clock = req_end - req_start;
        if(clock_par_min[(i-from)%num_thread] == 0 || 
                clock_par_min[(i-from)%num_thread] > tmp_clock)
            clock_par_min[(i-from)%num_thread] = tmp_clock;
        if(clock_par_max[(i-from)%num_thread] < tmp_clock)
            clock_par_max[(i-from)%num_thread] = tmp_clock;
        clock_par_req[(i-from)%num_thread] += tmp_clock;
    }
    clock_gettime(CLOCK_MONOTONIC, &et);
    clock_end = rdtsc();
    get_time_val += (1000LL * (et.tv_sec - st.tv_sec) + (et.tv_nsec - st.tv_nsec)/1000000.0f);
    clock_sum += clock_end - clock_start;

    cudaMemcpy(cuda_memory, user_get, request_size*(to-from)*sizeof(char), cudaMemcpyHostToDevice);
}

void get_data_rocks_w_req(char *user_get, char *cuda_memory, int memory_size, 
        int *gpu_request_list, int *host_request_list, int request_size, int stride, DB* db){
    int num_thread = thread;
    ReadOptions ro = ReadOptions();
    if(thread == 0) num_thread = 512;

    cudaMemcpy(host_request_list, gpu_request_list, sizeof(int)*stride, cudaMemcpyDeviceToHost);

    struct timespec st, et;
    unsigned long long clock_start, clock_end;
    clock_start = rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &st);
    #pragma omp parallel for num_threads(num_thread) schedule(dynamic)
    for(int i = 0; i < stride; ++i){
        unsigned long long req_start, req_end;
        char key[100];
        unsigned long long tmp_clock;
        std::string value;
        sprintf(key, "%d", host_request_list[i]);
        //db->Get(ReadOptions(), key, &value);
        req_start = rdtsc();
        db->Get(ro, key, &value);
        memcpy(user_get+i*request_size, value.data(), value.size());
        req_end = rdtsc();
        tmp_clock = req_end - req_start;
        if(clock_par_min[i%num_thread] == 0 || 
                clock_par_min[i%num_thread] > tmp_clock)
            clock_par_min[i%num_thread] = tmp_clock;
        if(clock_par_max[i%num_thread] < tmp_clock)
            clock_par_max[i%num_thread] = tmp_clock;
        clock_par_req[i%num_thread] += tmp_clock;
    }
    clock_gettime(CLOCK_MONOTONIC, &et);
    clock_end = rdtsc();
    get_time_val += (1000LL * (et.tv_sec - st.tv_sec) + (et.tv_nsec - st.tv_nsec)/1000000.0f);
    clock_sum += clock_end - clock_start;

    cudaMemcpy(cuda_memory, user_get, request_size*stride*sizeof(char), cudaMemcpyHostToDevice);
}

void get_data_rocks_c(char *user_get, char *cuda_memory, int from, int to,
        int memory_size, int request_size, rocksdb_t *db, rocksdb_readoptions_t *readoptions){ 
    int num_thread = thread;
    if(thread == 0) num_thread = 512;

    struct timespec st, et;
    unsigned long long clock_start, clock_end;
    clock_start = rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &st);
    #pragma omp parallel for num_threads(num_thread) schedule(dynamic)
    for(int i = from; i < to; ++i){
        unsigned long long req_start, req_end;
        char key[100];
        size_t len;
        char *err = NULL;
        char *val;
        unsigned long long tmp_clock;
        sprintf(key, "%d", i);
        req_start = rdtsc();
        val = rocksdb_get(db, readoptions, key, strlen(key), &len, &err);
        memcpy(user_get+(i-from)*request_size, val, len);
        free(val);
        req_end = rdtsc();
        tmp_clock = req_end - req_start;
        if(clock_par_min[(i-from)%num_thread] == 0 || 
                clock_par_min[(i-from)%num_thread] > tmp_clock)
            clock_par_min[(i-from)%num_thread] = tmp_clock;
        if(clock_par_max[(i-from)%num_thread] < tmp_clock)
            clock_par_max[(i-from)%num_thread] = tmp_clock;
        clock_par_req[(i-from)%num_thread] += tmp_clock;
    }
    clock_gettime(CLOCK_MONOTONIC, &et);
    clock_end = rdtsc();
    get_time_val += (1000LL * (et.tv_sec - st.tv_sec) + (et.tv_nsec - st.tv_nsec)/1000000.0f);
    clock_sum += clock_end - clock_start;

    cudaMemcpy(cuda_memory, user_get, request_size*(to-from)*sizeof(char), cudaMemcpyHostToDevice);

}

void get_data_rocks_c_w_req(char *user_get, char *cuda_memory, int memory_size, 
        int *gpu_request_list, int *host_request_list, int request_size, int stride, 
        rocksdb_t *db, rocksdb_readoptions_t *readoptions){
    int num_thread = thread;
    if(thread == 0) num_thread = 512;

    cudaMemcpy(host_request_list, gpu_request_list, sizeof(int)*stride, cudaMemcpyDeviceToHost);

    struct timespec st, et;
    unsigned long long clock_start, clock_end;
    clock_start = rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &st);
    #pragma omp parallel for num_threads(num_thread) schedule(dynamic)
    for(int i = 0; i < stride; ++i){
        unsigned long long req_start, req_end;
        char key[100];
        size_t len;
        char *err = NULL;
        char *val;
        unsigned long long tmp_clock;
        sprintf(key, "%d", host_request_list[i]);
        req_start = rdtsc();
        val = rocksdb_get(db, readoptions, key, strlen(key), &len, &err);
        memcpy(user_get+i*request_size, val, len);
        free(val);
        req_end = rdtsc();
        tmp_clock = req_end - req_start;
        if(clock_par_min[i%num_thread] == 0 || 
                clock_par_min[i%num_thread] > tmp_clock)
            clock_par_min[i%num_thread] = tmp_clock;
        if(clock_par_max[i%num_thread] < tmp_clock)
            clock_par_max[i%num_thread] = tmp_clock;
        clock_par_req[i%num_thread] += tmp_clock;
    }
    clock_gettime(CLOCK_MONOTONIC, &et);
    clock_end = rdtsc();
    get_time_val += (1000LL * (et.tv_sec - st.tv_sec) + (et.tv_nsec - st.tv_nsec)/1000000.0f);
    clock_sum += clock_end - clock_start;

    cudaMemcpy(cuda_memory, user_get, request_size*stride*sizeof(char), cudaMemcpyHostToDevice);
}

#else
void get_data_kvssd(char *user_get, char *cuda_memory, int from, int to,
        int memory_size, int request_size){
    int num_thread = thread;
    if(thread == 0) num_thread = 512;

    struct timespec st, et;
    unsigned long long clock_start, clock_end;
    clock_start = rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &st);
    #pragma omp parallel for num_threads(num_thread) schedule(dynamic)
    for(unsigned long i = from; i < to; ++i){
        unsigned long long req_start, req_end;
        unsigned long long tmp_clock;
        req_start = rdtsc();
        key_op(KEY_GET, i, (user_get+(i-from)*request_size), request_size);
        req_end = rdtsc();
        tmp_clock = req_end - req_start;
        if(clock_par_min[(i-from)%num_thread] == 0 || 
                clock_par_min[(i-from)%num_thread] > tmp_clock)
            clock_par_min[(i-from)%num_thread] = tmp_clock;
        if(clock_par_max[(i-from)%num_thread] < tmp_clock)
            clock_par_max[(i-from)%num_thread] = tmp_clock;
        clock_par_req[(i-from)%num_thread] += tmp_clock;
    }
    clock_gettime(CLOCK_MONOTONIC, &et);
    clock_end = rdtsc();
    get_time_val += (1000LL * (et.tv_sec - st.tv_sec) + (et.tv_nsec - st.tv_nsec)/1000000.0f);
    clock_sum += clock_end - clock_start;

    cudaMemcpy(cuda_memory, user_get, request_size*(to-from)*sizeof(char), cudaMemcpyHostToDevice);
}

void get_data_kvssd_w_req(char *user_get, char *cuda_memory, int memory_size, 
        int *gpu_request_list, int *host_request_list, int* whole_list, int request_size, int stride){
    int num_thread = thread;
    if(thread == 0) num_thread = 512;
    cudaMemcpy(host_request_list, gpu_request_list, sizeof(int)*stride, cudaMemcpyDeviceToHost);

    struct timespec st, et;
    unsigned long long clock_start, clock_end;
    clock_start = rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &st);

    #pragma omp parallel for num_threads(num_thread) schedule(dynamic)
    for(int i = 0; i < stride; ++i){
        unsigned long long req_start, req_end;
        unsigned long long tmp_clock;
        req_start = rdtsc();
        key_op(KEY_GET, host_request_list[i], (user_get+i*request_size), request_size);
        req_end = rdtsc();
        tmp_clock = req_end - req_start;
        if(clock_par_min[i%num_thread] == 0 || 
                clock_par_min[i%num_thread] > tmp_clock)
            clock_par_min[i%num_thread] = tmp_clock;
        if(clock_par_max[i%num_thread] < tmp_clock)
            clock_par_max[i%num_thread] = tmp_clock;
        clock_par_req[i%num_thread] += tmp_clock;
        //whole_list[host_request_list[i]]++;
    }
    clock_gettime(CLOCK_MONOTONIC, &et);
    clock_end = rdtsc();
    get_time_val += (1000LL * (et.tv_sec - st.tv_sec) + (et.tv_nsec - st.tv_nsec)/1000000.0f);
    clock_sum += clock_end - clock_start;

    cudaMemcpy(cuda_memory, user_get, request_size*stride*sizeof(char), cudaMemcpyHostToDevice);
}
#endif

int main(int argc, char **argv){
	int device = 0;
	int request_num=32*1024*128; // 32K * 128
    int workload=0, handle=1,call=1;
	unsigned int request_size=4096; // 4K
	int gpu_thread=256,gpu_block=256;
	int gpu_memory_size=(4*1024*1024); ///p2p mem :220MB
	int batch=32;

	printf("[DEBUG] Synthetic start\n");
	while(1){
		int c;
		c=getopt(argc,argv,"s:n:m:p:i:b:t:c:q:h");
		if(c==-1) break;
		switch(c){
			case 's':
				request_size=strtol(optarg,NULL,0);
				break;
			case 'n':
				request_num=strtol(optarg,NULL,0);
				break;
			case 'm':
				workload=strtol(optarg,NULL,0);
				break;
			case 'p':
				thread=strtol(optarg,NULL,0);
				break;
			case 'i':
				handle=strtol(optarg,NULL,0);
				break;
			case 'b':
				gpu_block=strtol(optarg,NULL,0);
				break;
			case 't':
				gpu_thread=strtol(optarg,NULL,0);
				break;
			case 'c':
				call=strtol(optarg,NULL,0);
				break;
			case 'q':
				batch=strtol(optarg,NULL,0);
				break;
			case 'h':
				printf("syntax: %s -s <request_size> -n <num_request> -m <workload:streaming-2,dynamic-4> -p <num_polling_thread> -i <process:one-0,batch_async-1,batch_sync-2> -q <batch_size> -b <gpu block> -t < gpu theads in block> -c <call:0-thread,1-warp,2-block> -h\n",argv[0]);
				exit(1);
				break;
			default:
				printf("ERROR: invalid option\n");
				exit(1);
				break;
		}
	}
#ifdef DATA_ON_ROCKS
#ifdef ROCKS_CXX
    #ifdef DIRECTIO
    char exp_target[] = "RocksCXX-Direct";
    #else
    char exp_target[] = "RocksCXX-Default";
    #endif
#else
    #ifdef DIRECTIO
    char exp_target[] = "RocksC-Direct";
    #else
    char exp_target[] = "RocksC-Default";
    #endif
#endif
#else
    char exp_target[] = "KVSSD";
#endif

    printf("[%16s] vertex: %d, mode: %s, thread: %d\n", exp_target, request_num, mode_name[workload/2-1], thread);
	//////////// GPU setting

	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));

	////////////////////////////////////////////////
	////////////////////p2p memory mapping
    /*
	void *dummy;
	ASSERTRT(cudaMalloc(&dummy,0));

	CUdeviceptr gpu_memory;
	char* cuda_memory;

	ASSERTDRV(cuMemAlloc(&gpu_memory, gpu_memory_size));
	ASSERTDRV(cuMemsetD8(gpu_memory, 0x00, gpu_memory_size));
	PRINT_TIMES("device ptr: 0x%lx \n",gpu_memory);
	unsigned int flag=1;
	ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, gpu_memory));
	struct I_KEY_PIN_PARAM pin;
	struct I_KEY_UNPIN_PARAM unpin;
	int pin_fd;

	pin_fd=open(D,O_RDWR);
	pin.addr=gpu_memory;
	pin.size=gpu_memory_size;
	pin.id=0;
	int ret=ioctl(pin_fd,I_KEY_PIN_BUF,&pin);
	if(ret!=0){printf("gpu pin error!\n"); return 0;}

	cuda_memory=(char*)gpu_memory;
    */

	////////// KV open
#ifdef DATA_ON_ROCKS
#ifndef ROCKS_CXX
	rocksdb_t *db;
	rocksdb_options_t *options = rocksdb_options_create();
	rocksdb_options_set_max_open_files(options, 524288);

	long cpus = sysconf(_SC_NPROCESSORS_ONLN);
	rocksdb_options_increase_parallelism(options, (int)cpus);
	rocksdb_options_optimize_level_style_compaction(options, 0);

	// Direct I/O options
	
		#ifdef DIRECTIO	
	rocksdb_options_set_use_direct_reads(options, true);
	rocksdb_options_set_use_direct_io_for_flush_and_compaction(options, true);
	

	// block cache off
	
	rocksdb_block_based_table_options_t* table = rocksdb_block_based_options_create();

	rocksdb_block_based_options_set_no_block_cache(table, true);
	rocksdb_block_based_options_set_cache_index_and_filter_blocks(table, false);

	rocksdb_options_set_block_based_table_factory(options, table);
		#endif
	/////////////

	rocksdb_options_set_create_if_missing(options, 1);
	char *err = NULL;
	size_t len;
	db = rocksdb_open(options, DBPath, &err);

	rocksdb_writeoptions_t *writeoptions = rocksdb_writeoptions_create();
	rocksdb_readoptions_t *readoptions = rocksdb_readoptions_create();

#elif defined(ROCKS_CXX)
	DB *db;
	Options options;

	options.IncreaseParallelism();
	options.OptimizeLevelStyleCompaction(0);
	options.create_if_missing = true;
	options.max_open_files = 524288;

    #ifdef DIRECTIO
	// Direct IO
	options.use_direct_reads = true;
	options.use_direct_io_for_flush_and_compaction = true;

	// Cache
	BlockBasedTableOptions table_options;
	table_options.no_block_cache = true;
	table_options.cache_index_and_filter_blocks = false;
	options.table_factory.reset(NewBlockBasedTableFactory(table_options));
    #endif

    options.statistics = CreateDBStatistics();
    options.statistics->set_stats_level(StatsLevel::kAll);

	Status s = DB::Open(options, DBPath, &db);

    // Perf statistics
    //SetPerfLevel(PerfLevel::kEnableTimeAndCPUTimeExceptForMutex);
	//get_perf_context()->Reset();
	//get_iostats_context()->Reset();

#endif

#else
	int key_ret = key_open(F_NAME);
#endif



    //////////////////////////////
    char *user_get = (char*)malloc(sizeof(char)*gpu_memory_size);
    char *cuda_memory;
    cudaMalloc(&cuda_memory, sizeof(char)*gpu_memory_size);

	cudaError_t error;

    /////////////////////////////////////// GPU kernel ///////////
    int stride = gpu_memory_size / request_size;
    gpu_block = stride / gpu_thread;

    /****************** request list make ********************/
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<unsigned long long> distribution(0, request_num-1);

	int *request_list,*cuda_request;
    int *whole_list;
	int i;
	request_list=(int*)malloc(sizeof(int)*stride);
    whole_list = (int*)calloc(request_num, sizeof(int));
	for(i=0;i<stride;i++){
		request_list[i]= distribution(generator);
	}
    ASSERTRT(cudaMalloc((void**)&cuda_request,sizeof(int)*stride));
	ASSERTRT(cudaMemcpy(cuda_request,request_list,sizeof(int)*stride,cudaMemcpyHostToDevice));

    /********* random ***********/
    curandState_t* states;
    cudaMalloc((void**)&states, sizeof(curandState_t)*stride);
    init_curand_state<<<gpu_block, gpu_thread>>>(states);


    /********* gpu kernel start *********/
    double time_before = _timestamp();
    unsigned long long clock_start = rdtsc(), clock_end;
    int iter = 3;
    double portion = 1;
    int round = request_num / stride;
    struct timespec start, end;
    double accum = 0;

    switch(workload){
        case 2:
            for(int i = 0; i < round; ++i){ 
                clock_gettime(CLOCK_MONOTONIC, &start);
#ifdef DATA_ON_ROCKS
    #ifdef ROCKS_CXX
                get_data_rocks(user_get, cuda_memory, i*stride, (i+1)*stride,
                        gpu_memory_size, request_size, db);
    #else
                get_data_rocks_c(user_get, cuda_memory, i*stride, (i+1)*stride,
                        gpu_memory_size, request_size, db, readoptions);
    #endif
#else
                get_data_kvssd(user_get, cuda_memory, i*stride, (i+1)*stride,
                        gpu_memory_size, request_size);
#endif
                clock_gettime(CLOCK_MONOTONIC, &end);
                accum += (1000LL * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1000000.0f);
                streaming_conv<<<gpu_block, gpu_thread>>>(
                //streaming_conv<<<1,1>>>(
                    cuda_memory, request_size, stride, iter, portion);
            }
            printf("Total get+memcpy: %lf sec,  avg. get+memcpy: %lf msec\n", accum/1000.0f, accum/(double)round);
            printf("Total get only: %lf sec, avg. get only: %lf msec\n", get_time_val/1000.0f, get_time_val/(double)round);
            break;
        case 4:
            portion = 0.90625;
            for(int i = 0; i < round; ++i){
                clock_gettime(CLOCK_MONOTONIC, &start);
#ifdef DATA_ON_ROCKS
    #ifdef ROCKS_CXX
                get_data_rocks_w_req(user_get, cuda_memory, gpu_memory_size, 
                        cuda_request, request_list, request_size, stride, db);
    #else
                get_data_rocks_c_w_req(user_get, cuda_memory, gpu_memory_size, 
                        cuda_request, request_list, request_size, stride, db, readoptions);

    #endif
#else
                get_data_kvssd_w_req(user_get, cuda_memory, gpu_memory_size, 
                        cuda_request, request_list, whole_list, request_size, stride);
#endif
                clock_gettime(CLOCK_MONOTONIC, &end);
                accum += (1000LL * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1000000.0f);
                dynamic_conv<<<gpu_block, gpu_thread>>>(
                    cuda_memory, request_size, cuda_request, stride, 
                    request_num, iter, portion, states);
            }
            printf("Total data get process: %lf sec,  avg. data get process: %lf msec\n", accum/1000.0f, accum/(double)round);
            printf("Total get only process: %lf sec,  avg. get only process: %lf msec\n", get_time_val/1000.0f, get_time_val/(double)round);
            break;
    }

    
    if(error){
    }
    double time_after = _timestamp();

    clock_end = rdtsc();
    double total_time = elapsed_time(time_before, time_after);
    unsigned long long clock_req_tot = 0;
    printf("stride: %d, gpu_thread: %d, gpu_block: %d, iter: %d, portion: %.2f | End total time: %8.3lf s\n\n", stride, gpu_thread, gpu_block, iter, portion, total_time);
    printf("tot clock: %llu\n", clock_end - clock_start);
    printf("req clock: %llu\n", clock_sum);
    for(int i = 0; i < thread; ++i){
        clock_req_tot += clock_par_req[i];
    }
    printf("partial clock sum: %llu\n", clock_req_tot);
    printf("min req: %llu\nmax req: %llu\n", *(std::min_element(clock_par_min, clock_par_min+thread)), *(std::max_element(clock_par_max, clock_par_max+thread)));
    printf("min req sum: %llu\nmax req sum: %llu\n", *(std::min_element(clock_par_req, clock_par_req+thread)), *(std::max_element(clock_par_req, clock_par_req+thread)));

    /*
    if(workload == 4){
        FILE* fp = fopen("whole_list_touch.txt", "w");
        int cnt = 0;
        for(int i = 0; i < request_num; ++i){
            fprintf(fp, "%d: %d\n", i, whole_list[i]);
            if(whole_list[i] == 0) cnt++;
        }
        fclose(fp);
        printf("not touched: %d\n", cnt);
    }
    */
#ifdef ROCKS_CXX
    char filename[100];
    sprintf(filename, "%s_%s_%d.txt", exp_target, mode_name[workload/2-1], thread);
    std::ofstream outfile(filename);
    //outfile << get_perf_context()->ToString() << std::endl;
    //outfile << get_iostats_context()->ToString() << std::endl;
    outfile << options.statistics->ToString() << std::endl;
    outfile.close();


	SetPerfLevel(PerfLevel::kDisable);
	//delete db;

#endif




    //////////////////////////////////////////////////////////////


    ////////////////////////////
    ///////// Finishing

    //unpinning

    // memory free
    CUDA_SAFE_CALL(cudaFree(cuda_memory));
	CUDA_SAFE_CALL(cudaFree(cuda_request));
	CUDA_SAFE_CALL(cudaFree(states));

	free(request_list);
    free(user_get);
    free(whole_list);

    // kv close

#ifdef DATA_ON_ROCKS
    #ifdef ROCKS_CXX
    delete db;
    #else
    rocksdb_writeoptions_destroy(writeoptions);
	rocksdb_readoptions_destroy(readoptions);
        #ifdef DIRECTIO
    	rocksdb_block_based_options_destroy(table);
        #endif
	rocksdb_options_destroy(options);
	rocksdb_close(db);
    #endif
#else
	key_close(F_NAME);
#endif

	cudaDeviceReset();
    return 0;
}
