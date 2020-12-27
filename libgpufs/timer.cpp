/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/

#ifndef TIMER
#define TIMER
#include<time.h>
/*
double _timestamp() {
	struct timespec tv;
	clock_gettime(CLOCK_MONOTONIC, &tv);
	return 1e9*tv.tv_sec + tv.tv_nsec;
}*/

#include <sys/time.h>

double _timestamp(){
              struct timeval tv;
                gettimeofday(&tv,0);
                return 1e6*tv.tv_sec+tv.tv_usec;
}
double elapsed_time(double start_time, double end_time) {
	double t;
	t=(end_time-start_time)/(1e6);
	return t;
}

		
#endif

