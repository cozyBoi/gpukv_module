#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <unistd.h>

#include "key_api.h"
#include "key-p2p.h"

#ifdef DATA_ON_ROCKS
#include "rocksdb/c.h"
#endif

#include "gardenia/timer.h"

#define TOTAL_VERTICES 1971281
//#define TOTAL_VERTICES 14
#define MAX_BUF 4096

#define KEY_PAGE_SIZE 4096
//using namespace rocksdb;

//const char DBPath[] = "/home/asknor/nvme/rocksdb";
const char DBPath[] = "/home/asknor/rocksdb2";

int main(int argc, char **argv){
	int tot_vert = TOTAL_VERTICES;
	if(argc > 2){
		tot_vert = atoi(argv[1]);
		// argv[2] = target folder
	}
	else{
		printf("Usage: %s [total vertices] [dataset folder]\n", argv[0]);
		return 0;
	}
	int i;
	char filename[101];
	char *buf[1];
	char tmp[MAX_BUF+1];
	if(posix_memalign((void**)&buf[0], KEY_PAGE_SIZE, KEY_PAGE_SIZE)){
		printf("can not allocate io payload buffer!\n");
		return 0;
	}
	memset(buf[0],0,KEY_PAGE_SIZE);

	printf("\tTotal vertices: %d\n",tot_vert);
	int req_size = KEY_PAGE_SIZE;
	Timer t;
#ifdef DATA_ON_ROCKS
	char key[100];
#else
	unsigned long key;
	int key_ret = key_open(F_NAME);
#endif
	int ret, n;

	// RocksDB setup
#ifdef DATA_ON_ROCKS
	rocksdb_t *db;
	rocksdb_options_t *options = rocksdb_options_create();
	rocksdb_options_set_max_open_files(options, 524288);
	printf("option created\n");

	long cpus = 4;
	rocksdb_options_increase_parallelism(options, (int)cpus);
	rocksdb_options_optimize_level_style_compaction(options, 0);

	rocksdb_options_set_create_if_missing(options, 1);

	char *err = NULL;
	size_t len;
	printf("rocksdb open\n");

	db = rocksdb_open(options, DBPath, &err);
	rocksdb_writeoptions_t *writeoptions = rocksdb_writeoptions_create();
	
	// Remove WAL
	rocksdb_writeoptions_disable_WAL(writeoptions, 1);

	rocksdb_writebatch_t *wb = rocksdb_writebatch_create();

	rocksdb_readoptions_t *readoptions = rocksdb_readoptions_create();
/*
	DB *db;
	Options options;

	options.IncreaseParallelism();
	options.OptimizeLevelStyleCompaction();
	options.create_if_missing = true;

	Status s = DB::Open(options, DBPath, &db);
*/
#endif

	t.Start();
	// vertex starts from 0 to TOTAL_VERTICES - 1
    printf("%02X %02X\n",(unsigned char)KEY_PUT, (unsigned char)KEY_GET);
	for(i = 0; i < tot_vert; ++i){
	//for(i = 1420000; i < tot_vert; ++i){
#ifdef DATA_ON_ROCKS
		sprintf(key, "%d", i);
#else
		key = i;
#endif
		sprintf(filename, "%s/%d.txt", argv[2], i);
		FILE* fp = fopen(filename, "rb");
		if(!fp){
			printf("[filename:%s] vertex %d doesn't exist\n",filename, i);
			//getchar();
			continue;
		}
		n = fread(tmp, sizeof(unsigned char), MAX_BUF, fp);
		fclose(fp);
		memcpy(buf[0], tmp, n);
		if(i < 10){
			printf("Before PUT\n");
			for(int j = 0; j < n; ++j)
				printf("%02X ", (unsigned char)buf[0][j]);
			printf("\n");
		}


		//printf("%d\n", n);
		//printf("%p %p %02x\n",buf, buf[0], buf[0][0]);
#ifdef DATA_ON_ROCKS
		//rocksdb_put(db, writeoptions, key, strlen(key), buf[0], KEY_PAGE_SIZE, &err);
		rocksdb_writebatch_put(wb, key, strlen(key), buf[0], KEY_PAGE_SIZE);
#else
		ret = key_op(KEY_PUT, key, buf[0], KEY_PAGE_SIZE);
		if(ret != KEY_PAGE_SIZE) fprintf(stderr, "[KEY:%d]PUT Error! ret=%d\n", key, ret);
#endif
		memset(buf[0], 0, KEY_PAGE_SIZE);
		

		/****** DATA validation *****/
#ifdef DATA_ON_ROCKS
		/*char *val = rocksdb_get(db, readoptions, key, strlen(key), &len, &err);
		memcpy(buf[0], val, len);
		free(val);
		if(len != KEY_PAGE_SIZE) {
			fprintf(stderr, "[key: %s] storing is inappropriate! (original size: %d, len: %d)\n", key, n , len);
			rocksdb_close(db);
			db = rocksdb_open(options, DBPath, &err);
			i--;
			continue;
		}*/
#endif

#ifndef DATA_ON_ROCKS
		if(i < 10){
#ifndef DATA_ON_ROCKS
			ret = key_op(KEY_GET, key, buf[0], KEY_PAGE_SIZE);
		    if(ret != KEY_PAGE_SIZE) fprintf(stderr, "[KEY:%d]GET Error! ret=%d\n", key, ret);
#else
			char *val = rocksdb_get(db, readoptions, key, strlen(key), &len, &err);
			memcpy(buf[0], val, len);
			free(val);
#endif
			//std::string s;
			//db->Get(ReadOptions(), std::to_string(key), &s);
			printf("KEY_GET %d\n", ret);
			for(int j = 0; j < n; ++j)
				printf("%02X ", (unsigned char)(buf[0][j]));
			printf("\n");
		}
#endif
#ifdef DATA_ON_ROCKS
		if((i != 0) && (i % 500000 == 0)) {
			rocksdb_write(db, writeoptions, wb, &err);
			rocksdb_compact_range(db, NULL, 0, NULL, 0);
			rocksdb_writebatch_clear(wb);
		}
#endif

		if(i % 10000 == 0) printf("%d finished\n", i);
	}
#ifdef DATA_ON_ROCKS
	rocksdb_write(db, writeoptions, wb, &err);
	rocksdb_writebatch_clear(wb);
#endif
	t.Stop();
	printf("\tPUT data for BFS elapsed time: %f ms.\n", t.Millisecs());

#ifndef DATA_ON_ROCKS
	for(i = 0; i < 10; ++i){
		key = i;
		ret = key_op(KEY_GET, key, buf[0], KEY_PAGE_SIZE);
		for(int j = 0 ; j < 20; ++j)
			printf("%02X ", (unsigned char)(buf[0][j]));
		printf("\n");
	}
#endif

#ifdef DATA_ON_ROCKS
	t.Start();
	rocksdb_compact_range(db, NULL, 0, NULL, 0);
	t.Stop();
	printf("\tTime for compacting data on RocksDB: %f ms.\n", t.Millisecs());
	printf("Write Check\n");
	for(i = 0; i < tot_vert; i+=200345){
		sprintf(key, "%d", i);
		char *val = rocksdb_get(db, readoptions, key, strlen(key), &len, &err);
		memcpy(buf[0], val, len);
		free(val);
		if(len != KEY_PAGE_SIZE) {
			printf("[key: %d] ERROR: len(%d) != KEY_PAGE_SIZE\n", i, len);
			continue;
		}
		for(int j = 0; j < 20; ++j)
			printf("%02X ", (unsigned char)(buf[0][j]));
		printf("\n");
	}

	rocksdb_writebatch_destroy(wb);
	rocksdb_writeoptions_destroy(writeoptions);
	rocksdb_readoptions_destroy(readoptions);
	rocksdb_options_destroy(options);
	rocksdb_close(db);
#else
	key_close(F_NAME);
#endif

	free(buf[0]);
	return 0;
}
