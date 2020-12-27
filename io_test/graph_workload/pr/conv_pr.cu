#include <stdio.h>

#include <omp.h>

#include "key-p2p.h"
#include "key_api.h"

#include "gpukv/util.cu.h"

#include "pr.cu.h"

#ifdef DATA_ON_ROCKS
//#include "rocksdb/db.h"
//#include "rocksdb/slice.h"
//#include "rocksdb/options.h"
#include "rocksdb/c.h"
#endif

#include "gardenia/timer.h"
#include "gardenia/worklistc.h"
#include "gardenia/common.h"

#define TOTAL_VERTICES 1971281

// MAX: 192x256 (p2p: 192*1204*1024 data: 4*1024 -> 48*1024 = 192*256)

const char DBPath[] = "/home/asknor/nvme/rocks_data";

char buffer[4097];

int thread = 0;

void get_graph_data(char *user_get, char *cuda_memory, int from, int to, int memory_size, int request_size){
	int num_thread = thread;
	if(thread == 0) num_thread = (to - from > 16384) ? 16384 : to - from;

	//Timer t_io;
    //t_io.Start();

	#pragma omp parallel for num_threads(num_thread) schedule(dynamic)
	for(int i = from; i < to; ++i){
		key_op(KEY_GET, i, (user_get+(i-from)*request_size), request_size);
	}
    //t_io.Stop();

	cudaMemcpy(cuda_memory, user_get, request_size*(to-from)*sizeof(char), cudaMemcpyHostToDevice);
    //printf("I/O time: %4.6f s\n", t_io.Seconds());
}

#ifdef DATA_ON_ROCKS
void get_graph_data_rocks(char *user_get, char *cuda_memory, int from, int to, int memory_size, int request_size,
		rocksdb_t *db, rocksdb_readoptions_t *readoptions){
	int num_thread = thread;
	if(thread == 0) num_thread = (to - from > 16384) ? 16384 : to - from;
	
    Timer t_io;
    //t_io.Start();
	#pragma omp parallel for num_threads(num_thread) schedule(dynamic)
	for(int i = from; i < to; ++i){
		char key[100];
		size_t len;
		char *err = NULL;
		char *val;
		sprintf(key, "%d", i);
		val = rocksdb_get(db,readoptions, key, strlen(key), &len, &err);
		//printf("key: %s len: %d\n", key, len);
		memcpy(user_get+(i-from)*request_size, val, len);
		free(val);
		//db->Get(ReadOptions(), std::to_string(queue->h_queue[i]), &value);
//		key_op(KEY_GET, queue->h_queue[i], (user_get+i*request_size), request_size);
		//memcpy(user_get+i*request_size, value.data(), value.size()*sizeof(char));
	}
    //t_io.Stop();

	cudaMemcpy(cuda_memory, user_get, request_size*(to-from)*sizeof(char), cudaMemcpyHostToDevice);
    //printf("I/O time: %4.6f s\n", t_io.Seconds());
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
#ifdef DIRECTIO
	printf("[rocks direct bfs] num I/O thread: %d\n", thread);
#else
	printf("[rocks        bfs] num I/O thread: %d\n", thread);
#endif
#else
	printf("[conventional bfs] num I/O thread: %d\n", thread);
#endif
	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));

	char *user_get = (char*)malloc(sizeof(char)*gpu_memory_size);
	char *cuda_memory;
	cudaMalloc(&cuda_memory, sizeof(char)*gpu_memory_size);

	int key_ret = key_open(F_NAME);

	/********** RocksDB setup ***********/
#ifdef DATA_ON_ROCKS
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

	rocksdb_options_set_create_if_missing(options, 1);

	char *err = NULL;
	size_t len;
	db = rocksdb_open(options, DBPath, &err);

	rocksdb_writeoptions_t *writeoptions = rocksdb_writeoptions_create();
	rocksdb_readoptions_t *readoptions = rocksdb_readoptions_create();
#endif


	/******************* bfs workload *******************/
	int max_req = 48*1024;
	int m = tot_vert;
	int stride;
	int *column_indices;
	int *degree;
	ScoreT *h_scores, *d_scores;
	ScoreT *d_contrib;
	ScoreT base_score = (1.0f - 0.85) / (float)m;
	float h_diff, *d_diff;

	int nthreads = BLOCK_SIZE, nblocks;
	nblocks = max_req / nthreads;


	h_scores = (ScoreT *)malloc(sizeof(ScoreT)*m);
	for(int i = 0; i < m; ++i)
		h_scores[i] = 1.0f/m;

	Timer t;
	t.Start();
	CUDA_SAFE_CALL(cudaMalloc((void**)&column_indices, nthreads*nblocks*128 * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&degree, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_contrib, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_diff, sizeof(float))); 
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, h_scores, m*sizeof(ScoreT), cudaMemcpyHostToDevice));

	for(stride = 0; stride < m ;stride += max_req){
		int end = (stride + max_req >= m) ? (m - 1) : (stride + max_req - 1);
#ifdef DATA_ON_ROCKS
		get_graph_data_rocks(user_get, cuda_memory, stride, end, gpu_memory_size, request_size, db, readoptions); 
#else
		get_graph_data(user_get, cuda_memory, stride, end, gpu_memory_size, request_size);
#endif
		set_degree<<<nblocks, nthreads>>>(cuda_memory, m, column_indices, stride, degree, request_size);
	}
	int iter = 0;
    Timer t_iter;
	do{
        t_iter.Start();
		iter++;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		nblocks = (m - 1) / nthreads + 1;
		contrib<<<nblocks, nthreads>>>(m, d_scores, degree, d_contrib);

		nblocks = max_req / nthreads;
		for(stride = 0; stride < m; stride += max_req){
			int end = (stride + max_req > m) ? (m) : (stride + max_req);
#ifdef DATA_ON_ROCKS
			get_graph_data_rocks(user_get, cuda_memory, stride, end, gpu_memory_size, request_size, db, readoptions); 
#else
			get_graph_data(user_get, cuda_memory, stride, end, gpu_memory_size, request_size);
#endif
			pull_fused<<<nblocks, nthreads>>>(cuda_memory, m, request_size, stride, column_indices, d_scores, d_contrib, d_diff, base_score);
		}
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
        t_iter.Stop();
        //printf("execution time for iter %3d: %4.6f sec\n", iter, t_iter.Seconds());
		//printf(" %2d    %f\n", iter, h_diff);
	}while(h_diff > EPSILON && iter < MAX_ITER);

	t.Stop();
	printf("Iter: %3d Calculation time: %f ms.(%f sec)\n",iter, t.Millisecs(), t.Seconds());

	key_close(F_NAME);

	CUDA_SAFE_CALL(cudaFree(column_indices));
	CUDA_SAFE_CALL(cudaFree(degree));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));

	CUDA_SAFE_CALL(cudaFree(cuda_memory));

	free(h_scores);
	free(user_get);

	//delete db;

#ifdef DATA_ON_ROCKS
	rocksdb_writeoptions_destroy(writeoptions);
	rocksdb_readoptions_destroy(readoptions);
#ifdef DIRECTIO
	rocksdb_block_based_options_destroy(table);
#endif
	rocksdb_options_destroy(options);
	rocksdb_close(db);
#endif

	cudaDeviceReset();
	return 0;
}
