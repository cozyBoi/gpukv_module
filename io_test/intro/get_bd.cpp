#include <cstdio>
#include <iostream>

#include <stdlib.h>
//#include <sys/time.h>
#include <time.h>

#include <unistd.h>
#include <stdlib.h>
//#include <stdio.h>
#include <fcntl.h>

#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <linux/types.h>
#include <linux/fs.h>


#include "rocksdb/iostats_context.h"
#include "rocksdb/perf_context.h"
#include "rocksdb/table.h"
#include "rocksdb/utilities/options_util.h"

#include "rocksdb/db.h"
#include "rocksdb/options.h"

using namespace rocksdb;

std::string DBPath = "/home/asknor/nvme/rocks_data";

int main(){
	srand(time(NULL));
	DB *db;
	Options options;

	options.IncreaseParallelism();
	options.OptimizeLevelStyleCompaction(0);
	options.create_if_missing = true;
	options.max_open_files = 524288;

	// Direct IO
    
	options.use_direct_reads = true;
	options.use_direct_io_for_flush_and_compaction = true;

	// Cache
	BlockBasedTableOptions table_options;
	table_options.no_block_cache = true;
	table_options.cache_index_and_filter_blocks = false;
	options.table_factory.reset(NewBlockBasedTableFactory(table_options));
    

	Status s = DB::Open(options, DBPath, &db);


	//SetPerfLevel(PerfLevel::kEnableTimeAndCPUTimeExceptForMutex);
	
    FILE* fp = fopen("../key_log.txt", "r");
	std::string value;
    unsigned long long accum = 0;
    double ac = 0;
	struct timespec st, et;
    //struct timespec accum={0};
		clock_gettime(CLOCK_MONOTONIC, &st);
	for(int i = 0; i < 1957027; i++){
		//int i_key = rand()%1971281;
		char key[20];
		struct timespec start, end, diff;

		//get_perf_context()->Reset();
		//get_iostats_context()->Reset();


        fscanf(fp, "%s", key);
        //sprintf(key, "%d", i_key);

		clock_gettime(CLOCK_MONOTONIC, &start);
		db->Get(ReadOptions(), key, &value);
		clock_gettime(CLOCK_MONOTONIC, &end);

        accum += 1000000000LL * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
        ac += (1000000000LL * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec))/1000.0f;


		//std::cout << "Key: " << key << std::endl;
		//std::cout << get_perf_context()->ToString() << std::endl;
		//std::cout << get_iostats_context()->ToString() << std::endl;
		//std::cout << std::endl << std::endl << std::endl;

	}
		clock_gettime(CLOCK_MONOTONIC, &et);

	//SetPerfLevel(PerfLevel::kDisable);
    //printf("get time(w/o read data): %lf us\n", (1000000000LL* accum.tv_sec + accum.tv_nsec)/10000/1000.0f);
    printf("get time(w/o read data): %lf us / %f us\n", ((double)accum)/1957057/1000.0f, ac/1957057);
    printf("In real: %lf us\n", (1000000000LL*(et.tv_sec - st.tv_sec) + (et.tv_nsec - st.tv_nsec))/1957057/1000.0f);

	int fd = open("/home/asknor/nvme/rocks_data/000006.sst", O_RDWR);
    unsigned int blknum = 0;

	struct timespec start, end, elap;

    clock_gettime(CLOCK_MONOTONIC, &start);
    int ret = ioctl(fd, FIBMAP, &blknum);
    clock_gettime(CLOCK_MONOTONIC, &end);

    elap.tv_sec = end.tv_sec - start.tv_sec;
    elap.tv_nsec = end.tv_nsec - start.tv_nsec;

    printf("fibmap time: %10f us (%5ld ns)\n",
            elap.tv_sec * 1000000 + (double)elap.tv_nsec/1000,
            elap.tv_sec * 1000000000UL + elap.tv_nsec
          );
	close(fd);


	delete db;

    return 0;
}
