// header??
#include "gpukv/fs_calls.cu.h"
#include "gpukv/util.cu.h"
#include <algorithm>

#define FILESIZE 786432 // 768KB
#define RGB_IDX  262144 // 256KB


#define WIDTH 512 
#define COVER 64

__device__ void
sobel(unsigned char* original, unsigned char *output){
//    int blk_idx = blockIdx.x;
//    int th_idx = threadIdx.x * 8 + threadIdx.y * 8;
    int th_x = threadIdx.x;
    int th_y = threadIdx.y;
//    int file_idx = blk_idx * FILESIZE;
    // file Input

    // gray scale processing
    for(int i = th_y * COVER; i < th_y * COVER + COVER; ++i){
        for(int j = th_x*COVER; j < th_x * COVER + COVER; ++j){
            original[i*WIDTH+j] = ((unsigned short)
                    original[i*WIDTH + j] +
                    original[RGB_IDX + i*WIDTH + j]+
                    original[RGB_IDX*2 + i*WIDTH + j])/3;
        }
    }
    __syncthreads();

    //sobel

    unsigned char pixs[9];
    short Horz, Vert, Sum;
    for(int y = th_y * COVER; y < th_y * COVER + COVER; ++y){
        for(int x = th_x*COVER; x < th_x * COVER + COVER; ++x){
            // pixs mapping
			if(x == 0){ //제일 왼쪽
                pixs[0] = 0;
                pixs[3] = 0;
                pixs[6] = 0;
            }
            else{ //아닌 경우
                if(y != 0)
                    pixs[0] = original[(y-1)*WIDTH + (x-1)]; 
				else pixs[0]=0;
                pixs[3] = original[y*WIDTH + (x-1)]; 
                if(y != WIDTH-1)
                    pixs[6] = original[(y+1)*WIDTH + (x-1)];
				else pixs[6]=0;
            }
           
            if(x == WIDTH-1){ //제일 오른쪽
                pixs[2] = 0;
                pixs[5] = 0;
                pixs[8] = 0;
            }
            else{ //아닌 경우
                if(y != 0)
                    pixs[2] = original[(y-1)*WIDTH + (x+1)]; 
				else pixs[2]=0;
                pixs[5] = original[y*WIDTH + (x+1)]; 
                if(y != WIDTH-1)
                    pixs[8] = original[(y+1)*WIDTH + (x+1)]; 
            }

            if(y == 0) //제일 위
                pixs[1] = 0;
            else
                pixs[1] = original[(y-1)*WIDTH + x];

            if(y == WIDTH-1) //제일 아래
                pixs[7] = 0;
            else
                pixs[7] = original[(y+1)*WIDTH + x];

            Horz = pixs[2] + 2*pixs[5] + pixs[8] - pixs[0] - 2*pixs[3] - pixs[6];
            Vert = pixs[0] + 2*pixs[1] + pixs[2] - pixs[6] - 2*pixs[7] - pixs[8];
            Sum = (short)((abs((int)Horz)+abs((int)Vert)));
            if(Sum < 0)
                Sum = 0;
            else if(Sum > 255)
                Sum = 255;
            Sum = (unsigned char)Sum;
            output[y*WIDTH + x] = Sum;
        }
    }
    __syncthreads();
	
    // output
}
