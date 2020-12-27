#include <stdio.h>
#include <sys/time.h>

#include <omp.h>

#include <time.h>

#include "key-p2p.h"
#include "key_api.h"

#include "gpukv/util.cu.h"

#include "bfs.cu.h"

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

using namespace rocksdb;
#endif

#include "gardenia/timer.h"
#include "gardenia/worklistc.h"
#include "gardenia/common.h"


#define TOTAL_VERTICES 1971281


//const char DBPath[] = "/home/asknor/nvme/rocksdb";
const char DBPath[] = "/home/asknor/nvme/rocks_data";

char buffer[4097];

int thread = 0;
int tot_req = 0;
int cumem_cnt = 0;
struct timeval cumem_time = {0,};
FILE* fp;
//struct timeval t_tmp[10000];
//struct timeval t_min = {999}, t_max = {0}, tot = {0};

void get_graph_data(char *user_get, char *cuda_memory, Worklist2 *queue, int memory_size, int request_size){
	int nitem = queue->nitems();
	int num_thread = thread;
	struct timeval start, end;
	if(thread == 0) num_thread = nitem;
	cudaMemcpy(queue->h_queue, queue->d_queue, nitem * sizeof(int), cudaMemcpyDeviceToHost);
	

	//gettimeofday(&start, NULL);
	#pragma omp parallel for num_threads(num_thread) schedule(dynamic)
	for(int i = 0; i < nitem; ++i){
		//gettimeofday(&start, NULL);
		key_op(KEY_GET, queue->h_queue[i], (user_get+i*request_size), request_size);
		//key_p2p_op(KEY_GET, queue->h_queue[i], i*request_size, request_size);
		//gettimeofday(&end, NULL);

		//get_time.tv_sec += end.tv_sec - start.tv_sec;
		//get_time.tv_usec += end.tv_usec - start.tv_usec;
		//tot_req++;
		/*printf("[req %7d] get time: %5f us\n", 
				tot_req, 
				(end.tv_sec - start.tv_sec)*1000000 + (double)(end.tv_usec - start.tv_usec)
				);*/
	}
	//gettimeofday(&end, NULL);
	//get_time.tv_sec += end.tv_sec - start.tv_sec;
	//get_time.tv_usec += end.tv_usec - start.tv_usec;


	gettimeofday(&start, NULL);
	cudaMemcpy(cuda_memory, user_get, nitem*request_size*sizeof(char), cudaMemcpyHostToDevice);
	gettimeofday(&end, NULL);
	cumem_cnt++;
	/*
	printf("[memcpy %3d] cumem time: %12f us\n", 
			cumem_cnt, 
			(end.tv_sec - start.tv_sec)*1000000 + (double)(end.tv_usec - start.tv_usec)
		  );
*/
	cumem_time.tv_sec += end.tv_sec - start.tv_sec;
	cumem_time.tv_usec += end.tv_usec - start.tv_usec;
	
}

#ifdef DATA_ON_ROCKS
void get_graph_data_rocks(char *user_get, char *cuda_memory, Worklist2 *queue, int memory_size, int request_size, rocksdb_t *db, rocksdb_readoptions_t *readoptions){
	int nitem = queue->nitems();
	int num_thread = thread;
	//struct timeval start[256],end[256];
	if(thread == 0) num_thread = nitem;
	cudaMemcpy(queue->h_queue, queue->d_queue, nitem * sizeof(int), cudaMemcpyDeviceToHost);
	

	tot_req += nitem;
#pragma omp parallel for num_threads(num_thread) schedule(dynamic)
	for(int i = 0; i < nitem; ++i){
		char key[100];
		size_t len;
		char *err = NULL;
		char *val;
		//struct timeval start, end;
		sprintf(key, "%d", queue->h_queue[i]);

		//gettimeofday(&start, NULL);
        //fprintf(fp,"%s\n",key);
		val = rocksdb_get(db,readoptions, key, strlen(key), &len, &err);
		//gettimeofday(&end, NULL);

		//t_tmp[i].tv_sec  = end.tv_sec - start.tv_sec;
		//t_tmp[i].tv_usec  = end.tv_usec - start.tv_usec;

		memcpy(user_get+i*request_size, val, len);
        //cudaMemcpy(cuda_memory+i*request_size, val, request_size, cudaMemcpyHostToDevice);
		free(val);
	}

	/*for(int i = 0; i < nitem; ++i){
		tot.tv_sec += t_tmp[i].tv_sec;
		tot.tv_usec += t_tmp[i].tv_usec;
		if((double)t_tmp[i].tv_sec*1000000 + (double)t_tmp[i].tv_usec > (double)t_max.tv_sec*1000000 + (double)t_max.tv_usec){
			t_max = t_tmp[i];
		}

		if((double)t_tmp[i].tv_sec*1000000 + (double)t_tmp[i].tv_usec < (double)t_min.tv_sec*1000000 + (double)t_min.tv_usec){
			t_min = t_tmp[i];
		}
	}*/
	//gettimeofday(&start, NULL);
	cudaMemcpy(cuda_memory, user_get, nitem*request_size*sizeof(char), cudaMemcpyHostToDevice);
	//gettimeofday(&end, NULL);
	//cumem_time.tv_sec += end.tv_sec - start.tv_sec;
	//cumem_time.tv_usec += end.tv_usec - start.tv_usec;
}

void get_graph_rocks_cpp(char *user_get, char *cuda_memory, Worklist2 *queue, int request_size, DB *db){
	int nitem = queue->nitems();
	int num_thread = thread;

	if(thread == 0) num_thread = nitem;
	cudaMemcpy(queue->h_queue, queue->d_queue, nitem*sizeof(int), cudaMemcpyDeviceToHost);

#pragma omp parallel for num_threads(num_thread) schedule(dynamic)
	for(int i = 0; i < nitem; ++i){
		char key[100];
	    std::string value;
		sprintf(key, "%d", queue->h_queue[i]);
		db->Get(ReadOptions(), key, &value);
		memcpy(user_get+i*request_size, value.data(), value.size());
	}
	cudaMemcpy(cuda_memory, user_get, nitem*request_size*sizeof(char), cudaMemcpyHostToDevice);
}
#endif

int main(int argc, char** argv){
	int device = 0;
	int gpu_thraed = 64, gpu_block = 256;
	int gpu_memory_size(192*1024*1024);
	int batch = 32;
	int tot_vert = TOTAL_VERTICES;
	unsigned int request_size = 4096;

	while(1){
		int c;
		c = getopt(argc, argv, "s:n:m:p:i:b:t:c:q:h");
		if(c == -1) break;
		switch(c){
			case 'p':
				thread=strtol(optarg, NULL, 0);
				break;
		}
	}

#ifdef DATA_ON_ROCKS
    #ifdef ROCKS_CXX
        #ifdef DIRECTIO
    printf("[rocks perf CXX D] num I/O thread: %d\n", thread);
        #else
    printf("[rocks perf CXX  ] num I/O thread: %d\n", thread);
        #endif
    #else
        #ifdef DIRECTIO
	printf("[rocks direct I/O] num I/O thread: %d\n", thread);
        #else
	printf("[rocks No Direct ] num I/O thread: %d\n", thread);
        #endif
    #endif
#else
	printf("[conventional bfs] num I/O thread: %d\n", thread);
#endif
    //fp = fopen("key_log.txt","w");
	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));

	////////// P2P
	/*
	int p2p_size = (192*1024*1024);
	CUdeviceptr gpu_memory;
	char* cuda_memory;

	void *dummy;
	ASSERTRT(cudaMalloc(&dummy, 0));

	ASSERTDRV(cuMemAlloc(&gpu_memory, p2p_size));
	ASSERTDRV(cuMemsetD8(gpu_memory, 0x00, p2p_size));

	unsigned int flag = 1;
	ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, gpu_memory));

	struct I_KEY_PIN_PARAM pin;
	struct I_KEY_UNPIN_PARAM unpin;
	int pin_fd;

	pin_fd = open(D, O_RDWR);
	pin.addr = gpu_memory;
	pin.size = p2p_size;
	pin.id = 0;
	int ret = ioctl(pin_fd, I_KEY_PIN_BUF, &pin);
	if(ret){
		printf("gpu pin error!\n");
		return 0;
	}

	cuda_memory = (char*)gpu_memory;
	*/
	////////////////////////////////////////////////////


	char *user_get = (char*)malloc(sizeof(char)*gpu_memory_size);
	char *cuda_memory;
	cudaMalloc(&cuda_memory, sizeof(char)*gpu_memory_size);

	Timer t;

	/********** RocksDB setup ***********/
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

	Status s = DB::Open(options, DBPath, &db);


	//SetPerfLevel(PerfLevel::kEnableTimeAndCPUTimeExceptForMutex);
	//get_perf_context()->Reset();
	//get_iostats_context()->Reset();
	#endif
#else
	int key_ret = key_open(F_NAME);
#endif


	/******************* bfs workload *******************/
	int m = tot_vert;
	int *column_indices;
	DistT *d_dist;
	DistT *h_dist;

	int src = 0;
	Worklist2 queue1(m), queue2(m);
	Worklist2 *in_queue = &queue1, *out_queue = &queue2;
	int *queue_elems = (int*)malloc(sizeof(int)*m);
	int nthreads = BLOCK_SIZE, nblocks;
	int nitems;


	h_dist = (DistT *)malloc(sizeof(DistT)*m);
	for(int i = 0; i < m; ++i)
		h_dist[i] = MYINFINITY;

	t.Start();
	insert<<<1, 1>>>(src, *in_queue);
	CUDA_SAFE_CALL(cudaMalloc((void**)&column_indices, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m*sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[src], &zero, sizeof(zero), cudaMemcpyHostToDevice));

	int iter = 1;
	do{
		nblocks = (nitems - 1) / nthreads + 1;

#ifdef DATA_ON_ROCKS
	#ifdef ROCKS_CXX
		get_graph_rocks_cpp(user_get, cuda_memory, in_queue, request_size, db);
	#else
		get_graph_data_rocks(user_get, cuda_memory, in_queue, gpu_memory_size, request_size, db, readoptions);
	#endif
#else
		get_graph_data(user_get, cuda_memory, in_queue, gpu_memory_size, request_size);
#endif

		bfs_no_gpukv<<<nblocks, nthreads>>>(cuda_memory, column_indices, request_size, d_dist, *in_queue, *out_queue);

		nitems = out_queue->nitems();
		//printf("Round %d finished! (next: %d)\n", iter, nitems);

		Worklist2 *tmp = in_queue;
		in_queue = out_queue;
		out_queue = tmp;
		out_queue->reset();
		iter++;
	}while(nitems > 0);

	t.Stop();
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));

	// Check BFS results
	/*for(int i = 0; i < (m > 100 ? 20 : m); i++){
		printf("dist[%d] = %u\n", i, h_dist[i]);
	}*/

	printf("Calculation time: %f ms.(%f sec)\n", t.Millisecs(), t.Seconds());
    //fclose(fp);
#ifdef ROCKS_CXX
	std::ofstream outfile("RocksPerf.txt", "w");
	outfile << get_perf_context()->ToString() << std::endl;
	outfile << get_iostats_context()->ToString() << std::endl;
	outfile.close();

	SetPerfLevel(PerfLevel::kDisable);
	delete db;
#endif

	/*printf("Avg. Rocks get time(total %d reqs): %f ms.(%f sec)\n", 
			tot_req,
			(tot.tv_sec*1000 + (double)tot.tv_usec/1000)/(double)tot_req,
			(tot.tv_sec + (double)tot.tv_usec/1e6)/(double)tot_req
			);

	printf("Min: %f ms. (%f sec)\n",
			t_min.tv_sec*1000 + (double)t_min.tv_usec/1000,
			t_min.tv_sec + (double)t_min.tv_usec/1e6
		  );
	printf("Min: %f ms. (%f sec)\n",
			t_max.tv_sec*1000 + (double)t_max.tv_usec/1000,
			t_max.tv_sec + (double)t_max.tv_usec/1e6
		  );*/

	
	printf("cudamem avg time(total %d iter): %f ms.(%f sec)\n",
			iter,
			(cumem_time.tv_sec * 1000 + (double)cumem_time.tv_usec / 1000)/(double)cumem_cnt,
			(cumem_time.tv_sec + (double)cumem_time.tv_usec / 1e6)/(double)cumem_cnt
		  );
		  


#ifndef DATA_ON_ROCKS
	//////////////////// P2P
	/*
	ASSERT_EQ(ioctl(pin_fd, I_KEY_UNPIN_BUF, &unpin), 0);
	ASSERTDRV(cuMemFree(gpu_memory));
	close(pin_fd);
	*/
	//////////////////////
	key_close(F_NAME);
#endif



	CUDA_SAFE_CALL(cudaFree(column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	CUDA_SAFE_CALL(cudaFree(cuda_memory));

	free(h_dist);
	free(queue_elems);
	free(user_get);

	//delete db;
#ifdef DATA_ON_ROCKS
#ifndef ROCKS_CXX
	rocksdb_writeoptions_destroy(writeoptions);
	rocksdb_readoptions_destroy(readoptions);
#ifdef DIRECTIO
	rocksdb_block_based_options_destroy(table);
#endif
	rocksdb_options_destroy(options);
	rocksdb_close(db);
#endif
#endif

	cudaDeviceReset();
	return 0;
}
