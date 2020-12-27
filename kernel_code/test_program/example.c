#include<stdio.h>
#include"key_api.h"

#define SIZE 4096*256
int main()
{
	int i;
	int ret;
	unsigned long key=100;
	char buf[SIZE+1];
	char *get_p=NULL;
	for(i=0;i<SIZE;i++){
		buf[i]='A'+(i%26);
	}
	buf[SIZE]='\0';
	ret=key_open("/dev/nvme0n1"); //device path ,success if ret>0
	ret=key_put(key,buf); //ret : write length
	key=200;
	ret=key_get(key,&get_p); //ret : read length
//	printf("%s",get_p);
	ret=key_close("/dev/nvme0n1"); //success if ret>0
	free(get_p);
	return 0;
}
