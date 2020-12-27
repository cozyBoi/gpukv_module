#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <unistd.h>

#include "key_api.h"
#include "key-p2p.h"

#include "rocksdb/c.h"

#include "gardenia/timer.h"

#define TOTAL_VERTICES 1971281
//#define TOTAL_VERTICES 14
#define MAX_BUF 2048

#define KEY_PAGE_SIZE 4096
//using namespace rocksdb;

const char DBPath[] = "/home/asknor/nvme/rocksdb";

int main(int argc, char **argv){
	int tot_vert = TOTAL_VERTICES;
	if(argc > 2){
		tot_vert = atoi(argv[1]);
		// argv[2] = target folder
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
	int key_ret = key_open(F_NAME);
	int req_size = KEY_PAGE_SIZE;
	Timer t;
	//unsigned long key;
	char key[100];
	int ret, n;

	// RocksDB setup
	rocksdb_t *db;
	rocksdb_options_t *options = rocksdb_options_create();
	printf("option created\n");

	long cpus = 1;
	rocksdb_options_increase_parallelism(options, (int)cpus);
	rocksdb_options_optimize_level_style_compaction(options, 0);

	rocksdb_options_set_create_if_missing(options, 1);

	char *err = NULL;
	size_t len;
	printf("rocksdb open\n");

	db = rocksdb_open(options, DBPath, &err);
	rocksdb_writeoptions_t *writeoptions = rocksdb_writeoptions_create();
	rocksdb_readoptions_t *readoptions = rocksdb_readoptions_create();
/*
	DB *db;
	Options options;

	options.IncreaseParallelism();
	options.OptimizeLevelStyleCompaction();
	options.create_if_missing = true;

	Status s = DB::Open(options, DBPath, &db);
*/
	t.Start();
	// vertex starts from 0 to TOTAL_VERTICES - 1
	for(i = 0; i < tot_vert; ++i){
		sprintf(key, "%d", i);
		sprintf(filename, "%s/%d.txt", argv[2], i);
		FILE* fp = fopen(filename, "rb");
		if(!fp){
			printf("[filename:%s] vertex %d doesn't exist\n",filename, i);
			//getchar();
			continue;
		}
		n = fread(tmp, sizeof(char), MAX_BUF, fp);
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
		//ret = key_op(KEY_PUT, key, buf[0], KEY_PAGE_SIZE);
		rocksdb_put(db, writeoptions, key, strlen(key), buf[0], KEY_PAGE_SIZE, &err);
		memset(buf[0], 0, KEY_PAGE_SIZE);
		char *val = rocksdb_get(db, readoptions, key, strlen(key), &len, &err);
		memcpy(buf[0], val, len);
		free(val);
		if(len != KEY_PAGE_SIZE) {
			fprintf(stderr, "[key: %s] storing is inappropriate! (original size: %d, len: %d)\n", key, n , len);
			rocksdb_close(db);
			db = rocksdb_open(options, DBPath, &err);
			i--;
			continue;
		}
		//db->Put(WriteOptions(), std::to_string(key), buf[0]);
		//if(ret != KEY_PAGE_SIZE) fprintf(stderr, "[KEY_OP]Error! ret=%d\n", ret);
		if(i < 10){
			//ret = key_op(KEY_GET, key, buf[0], KEY_PAGE_SIZE);
			//std::string s;
			//db->Get(ReadOptions(), std::to_string(key), &s);
			printf("KEY_GET\n");
			for(int j = 0; j < n; ++j)
				printf("%02X ", (unsigned char)(buf[0][j]));
			printf("\n");
		}

		if(i % 10000 == 0) printf("%d finished\n", i);
	}
	t.Stop();
	printf("\tPUT data for BFS elapsed time: %f ms.\n", t.Millisecs());
	rocksdb_writeoptions_destroy(writeoptions);
	rocksdb_readoptions_destroy(readoptions);
	rocksdb_options_destroy(options);
	rocksdb_close(db);
	key_close(F_NAME);
	return 0;
}
