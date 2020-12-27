#include <cstdio>
#include <string>

#include "key_api.h"
#include "key-p2p.h"

#include "gpukv/util.cu.h"
/*
#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"
*/
#include "rocksdb/c.h"
#include "gardenia/timer.h"

//using namespace rocksdb;

//std::string DBPath = "/home/asknor/nvme/rocksdb";
const char* DBPath = "/home/asknor/rocks_data";

int main(){
	int device = 0;
	char *cudaMemory;
	int gpu_memory_size = 192*1024*1024;

	rocksdb_t *db;
	rocksdb_options_t *options = rocksdb_options_create();
	rocksdb_options_set_max_open_files(options, 524288);
	printf("option created\n");

	long cpus = 2;
	rocksdb_options_increase_parallelism(options, (int)cpus);
	rocksdb_options_optimize_level_style_compaction(options, 0);

	rocksdb_options_set_create_if_missing(options, 1);

	char *err = NULL;
	size_t len;
	char key[20];
	int i, tot_vert = 1971281;
	printf("rocksdb open\n");

	db = rocksdb_open(options, DBPath, &err);
	rocksdb_writeoptions_t *writeoptions = rocksdb_writeoptions_create();
	
	// Remove WAL
	rocksdb_writeoptions_disable_WAL(writeoptions, 1);

	rocksdb_writebatch_t *wb = rocksdb_writebatch_create();

	rocksdb_readoptions_t *readoptions = rocksdb_readoptions_create();

	for(i = 0; i < tot_vert; i++){
		sprintf(key, "%d", i);
		char *val = rocksdb_get(db, readoptions, key, strlen(key), &len, &err);
		if(len != KEY_PAGE_SIZE) printf("[key: %d] WARN: len != KEY_PAGE_SIZE (len: %d)\n",i, len);
		if(err){
			fprintf(stderr,"[key: %d] ERROR: %s\n",i, err);
		}
		free(val);
	}

	rocksdb_writebatch_destroy(wb);
	rocksdb_writeoptions_destroy(writeoptions);
	rocksdb_readoptions_destroy(readoptions);
	rocksdb_options_destroy(options);
	rocksdb_close(db);
/*
	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));

	cudaMalloc(&cudaMemory, sizeof(char)*gpu_memory_size);
*/
	// RocksDB setup
/*
	DB *db;
	Options options;

	options.IncreaseParallelism();
	options.OptimizeLevelStyleCompaction();
	options.create_if_missing = true;
	options.max_open_files = 524288 

	Status s = DB::Open(options, DBPath, &db);
	if(!s.ok()) cerr << s.ToString() << endl;
	assert(s.ok());

	// SST file open failure problem due to open file restriction
	//s = db->SetDBOptions({{"max_open_files", "524288"}});
	assert(s.ok());

	// Buffer setting
	int buffer_size=KEY_PAGE_SIZE;
	char *buffer[10];
	for(int i = 0; i < 10; ++i){
		if(posix_memalign((void**)&buffer[i],KEY_PAGE_SIZE,buffer_size)){
			printf("can not allocate io payload buffer!\n");
			return 0;
		}
	}
*/
	// Put value
	/*
	for(int i = 0; i < 25600; ++i){
		char num = rand() % 10 + '0';
		sprintf(key,"%d", i);
		memset(buffer[i%10], num, buffer_size);
		db->Put(WriteOptions(), key, buffer[i%10]);
	}
	*/

	/*
	// Get -> Memcpy
	std::string value;
	Timer t;
	for(int i = 0; i < 1927181; i++){
		sprintf(key, "%d", i);
		//t.Start();
		db->Get(ReadOptions(), key, &value);
		//t.Stop();
		//printf("Get(%4d): %.2f us,\t",i, t.Microsecs());
		//t.Start();
		//cudaMemcpy(cudaMemory+buffer_size*i, value.data(), value.size()*sizeof(char), cudaMemcpyHostToDevice);
		//t.Stop();
		//printf("cuda memcpy: %.2f us (size: %lu)\n", t.Microsecs(), value.size());
	}
	delete db;
	*/
	return 0;
}
