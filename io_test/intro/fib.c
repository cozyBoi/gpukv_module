#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>

#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <linux/types.h>
#include <linux/fs.h>

#include <time.h>

int main(){
	int fd = open("/home/asknor/nvme/rocks_data/000006.sst", O_RDWR);
	unsigned int blknum = 0;

	//struct timeval start, end, elap;
	struct timespec start, end, elap;
	srand(time(NULL));

	for(int i = 0; i < 100; ++i){
		blknum = rand() % 100000;
		//gettimeofday(&start, NULL);
		clock_gettime(CLOCK_MONOTONIC, &start);
		int ret = ioctl(fd, FIBMAP, &blknum);
		clock_gettime(CLOCK_MONOTONIC, &end);
		//gettimeofday(&end, NULL);

		elap.tv_sec = end.tv_sec - start.tv_sec;
		elap.tv_nsec = end.tv_nsec - start.tv_nsec;
		printf("ret: %d, blknum: %7u ", ret, blknum);

		printf("fibmap time: %10f us (%5ld ns)\n",
				elap.tv_sec * 1000000 + (double)elap.tv_nsec/1000,
				elap.tv_sec * 1000000000UL + elap.tv_nsec
			  );
	}
	close(fd);

	return 0;

}

