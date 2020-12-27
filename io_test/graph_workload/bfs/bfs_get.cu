#include <stdio.h>
#include <stdlib.h>

#include "key_api.h"
#include "key-p2p.h"
#include "gardenia/timer.h"

#define TOTAL_VERTICES 1971279
#define MAX_BUF 2048

int main(int argc, char **argv){
	int tot_vert = TOTAL_VERTICES;
	if(argc == 2){
		tot_vert = atoi(argv[1]);
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
	unsigned long key;
	int ret, n;
	t.Start();
	for(i = 0; i < TOTAL_VERTICES; i+=100001){
		key = i + 10000;
		//n = fread(tmp, sizeof(char), MAX_BUF, fp);
		memcpy(buf[0], tmp, n);
		//printf("%d\n", n);
		//printf("%p %p %02x\n",buf, buf[0], buf[0][0]);
		ret = key_op(KEY_GET, key, buf[0], KEY_PAGE_SIZE);
		for(int i = 4095; i > 4065; --i)
			printf("%02x ",buf[0][i]);
		printf("\n");
		if(ret != KEY_PAGE_SIZE) fprintf(stderr, "[KEY_OP]Error! ret=%d\n", ret);
		//if(i % 10000 == 0) printf("%d finished\n", i);
	}
	t.Stop();
	printf("\tPUT data for BFS elapsed time: %f ms.\n", t.Millisecs());
	key_close(F_NAME);
	return 0;
}
