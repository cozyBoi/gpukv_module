/*
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */

/*
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */

#ifndef HOST_LOOP_CPP
#define HOST_LOOP_CPP

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <pthread.h>
#include <iostream>
#include <iomanip>
#include<stdlib.h>

#include <nvToolsExt.h>

#include "fs_initializer.cu.h"
#include "key_api.h"
#include "key-p2p.h"

//#define TRACE

#ifdef TRACE

#define PRINT_TRACE(...) //fprintf(stderr, __VA_ARGS__);

#else

#define PRINT_TRACE(...)

#endif

#ifdef TIMING_STATS

#define PRINT_TIMES(...) //fprintf(stderr, __VA_ARGS__);

#else

#define PRINT_TIMES(...)

#endif


#define RW_HOST_WORKERS 10 //65 ///possible
//#define BATCH_LIMIT 128 ////may be changed?
#define RW_SLOTS_PER_WORKER(a) (RW_IPC_SIZE / a) //

double CpyTime[RW_HOST_WORKERS] = {0};
int CpyCount[RW_HOST_WORKERS] = {0};
unsigned long CpySize[RW_HOST_WORKERS] = {0};

int worker=1;
int batch_size=128;
volatile int done = 0;


pthread_mutex_t task_lock;
pthread_cond_t  task_cond;
pthread_t rwLoopTasksIDs[RW_HOST_WORKERS];
TaskData rwLoopTasksData[RW_HOST_WORKERS];

//cudaEvent_t events[RW_HOST_WORKERS];
/*
typedef struct __event_buf{
    int f,r;
    pthread_spinlock_t spinlock;
}Event_buf;

Event_buf event_list[RW_HOST_WORKERS];

*/
#define BUF_NEXT(a,num) ((a+1)%num)

static __inline__ unsigned long long hl_rdtsc(void){
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc": "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo | ( ((unsigned long long)hi) << 32) );
}

unsigned long long clock_polling = 0;
unsigned long long clock_req = 0;


void* complete_async(void* param) //async
{
    
    TaskData* taskData = (TaskData*)param;

    int pin_fd = taskData->id;
    int ret=ioctl(pin_fd,I_KEY_ASYNC,(unsigned long)0);
    if(ret!=0)
        printf("ASYNC error!\n");
    return NULL;
}

void start_complete_async(int pin_fd){
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);

    pthread_t complete_task;

    TaskData complete_data;
    complete_data.id = pin_fd;

    pthread_create(&complete_task, &at, complete_async, &complete_data);
    pthread_attr_destroy(&at);
}

void end_complete_async(int pin_fd){
    int ret = ioctl(pin_fd, I_KEY_ASYNC_CLOSE, (unsigned long)0);
    if(ret != 0)
        printf("ASYNC_CLOSE error!\n");
    printf("[DEBUG:host_loop]complete thread destroyed\n");
}



void* rw_task( void* param )/////just one process per one request
{
    TaskData* taskData = (TaskData*)param;

    int id = taskData->id;
    volatile GPUGlobals* globals = taskData->gpuGlobals;

    int num=RW_SLOTS_PER_WORKER(worker);

    int firstSlot = id * num;

    volatile char* push_list;
    
    //double temp_time;

    int num_request=0;

    char* user_put=(globals->user_put);
    char* cuda_put=(globals->cuda_put);
    cudaStream_t stream=globals->streamMgr->memStream[id];
    cudaEvent_t event;
    CUDA_SAFE_CALL(cudaEventCreate(&event));

    push_list=&(globals->cpu_ipcRWQueue->push_list[firstSlot]);
    while( !done )
    {
        if(id==0){//first thread
            volatile cudaError_t cuda_status = cudaStreamQuery( globals->streamMgr->kernelStream );
            if ( cudaErrorNotReady != cuda_status )
            {
                done = 1;
                continue;
            }
        }
        for (int i = 0; i < num; i++)
        {
            if ((push_list[i]) == CPU_IPC_PENDING) {
                (push_list[i]) = CPU_IPC_IN_PROCESS;
                //  PRINT_TRACE("Handle request in worker: %d, scratch id: %d, scratch offset: %ld, request id: %d, file offset: %ld\n",id, scratchIdx, scratchSize, firstSlot + i, e->file_offset);
                volatile CPU_IPC_RW_Entry* e = &globals->cpu_ipcRWQueue->entries[firstSlot + i];
                unsigned long key, offset;
                unsigned int size, mode;
                int return_value = 1;//0;
                key = e->key;
                offset = e->offset;
                size = e->size;
                mode = e->mode;
                //////////////////////
//              temp_time = _timestamp();
                //////////////////////
                switch (mode) {
                    case GET:
                        {
                            //                  PRINT_TRACE("GET key: %lu, offset: %lu, size : %u,  request id: %d, worker: %d\n", key, offset, size,  firstSlot + i, id);
                            return_value = key_p2p_op(GET,key, offset, size);
                            break;
                        }
                    case PUT:
                        {
                            //                  PRINT_TRACE("PUT key: %lu, offset: %lu, size : %u,  request id: %d, worker: %d\n", key, offset, size,  firstSlot + i, id);
                            CUDA_SAFE_CALL(cudaMemcpyAsync(user_put+offset,cuda_put+offset,size,cudaMemcpyDeviceToHost,stream));
                            //              printf("put_user : ");
                            //              for(int j=0;j<10;j++) printf("%c",user_put[offset+j]);
                            //              printf("\n");
                            CUDA_SAFE_CALL(cudaEventRecord(event,stream));
                            CUDA_SAFE_CALL(cudaEventSynchronize(event));
                            return_value = key_op(PUT,key,(user_put+offset), size);

                            break;
                        }
                    default:
                        assert(NULL);
                }
            
                assert(return_value >= 0); ///È¤½Ã³ª.....
                e->return_value = return_value;
                __sync_synchronize();
                (push_list[i]) = CPU_IPC_COMPLETE;
                __sync_synchronize();
/////////////////////////////////////////
                CpyCount[id]++;
                CpySize[id] += size;
//              CpyTime[id] += _timestamp() - temp_time;

                //////////////////////////////////////////////
                num_request++;
//              printf("num_request: %d \n",num_request);
            }
        }

    }
    CUDA_SAFE_CALL(cudaEventDestroy(event));

    return NULL;
}

typedef struct _user_list {
    unsigned long key;
    unsigned long offset;
    unsigned int size;
    unsigned int mode;
    int num;
    unsigned long result_offset;
    unsigned long complete_offset;
}User_list;

typedef struct _put_list{
    int *event_list;
    cudaEvent_t* events;
    volatile int f,r;
    int size;
}Put_list;

Put_list put_list;

void* rw_task_batch_sync(void* param) //async
{
    
    TaskData* taskData = (TaskData*)param;

    int id = taskData->id;
    int i;
    volatile GPUGlobals* globals = taskData->gpuGlobals;
    int num=RW_SLOTS_PER_WORKER(worker);

    int firstSlot = id * num;

    volatile char* push_list;
    
    //double temp_time;


    char* user_put=(globals->user_put);
    char* cuda_put=(globals->cuda_put);

    push_list=&(globals->cpu_ipcRWQueue->push_list[firstSlot]);
    
    User_list* request_buf;
    request_buf=(User_list*)malloc(sizeof(User_list)*batch_size);
    
    int put_num=num+1;  
    int *event_list;
    cudaEvent_t *events;
    events=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*put_num);
    event_list=(int*)malloc(sizeof(int)*put_num);
    for(i=0;i<put_num;i++){
        CUDA_SAFE_CALL(cudaEventCreate(&events[i]));
    }
    cudaStream_t stream=globals->streamMgr->memStream[id];

    int f=0,r=0; //for event_circular queue
    int j;
    
    int cnt = 0;

    while (!done)
    {
        if(id==0){//first thread
            volatile cudaError_t cuda_status = cudaStreamQuery( globals->streamMgr->kernelStream );
            if ( cudaErrorNotReady != cuda_status )
            {
                done = 1;
                continue;
            }
        }

        for (int i = 0; i < num; i++)
        {
            if ((push_list[i]) == CPU_IPC_PENDING) {
                //  PRINT_TRACE("Handle request in worker: %d, scratch id: %d, scratch offset: %ld, request id: %d, file offset: %ld\n",id, scratchIdx, scratchSize, firstSlot + i, e->file_offset);
                (push_list[i]) = CPU_IPC_IN_PROCESS;
                volatile CPU_IPC_RW_Entry* e = &globals->cpu_ipcRWQueue->entries[firstSlot + i];
                if(e->mode==GET){
                    request_buf[cnt].offset = e->offset;
                }
                else{
                    unsigned long offset;
                    unsigned int size;

                    offset=e->offset;
                    size=e->size;
                    CUDA_SAFE_CALL(cudaMemcpyAsync(user_put+offset,cuda_put+offset,size,cudaMemcpyDeviceToHost,stream));
                    CUDA_SAFE_CALL(cudaEventRecord(events[r],stream));
                    event_list[r]=i;
                    r=BUF_NEXT(r,put_num);
                    continue;
                }
                request_buf[cnt].key = e->key;
                request_buf[cnt].size = e->size;
                request_buf[cnt].mode = GET;
                request_buf[cnt].num = firstSlot + i;
                //////////////////////
                CpySize[id] += request_buf[cnt].size;

                cnt++;
                if (cnt >= batch_size) {
                    CpyCount[id]+=cnt;
//                  temp_time = _timestamp();
    //              printf("key_batch_start : %d\n",cnt);
                    if(key_batch_op((unsigned long)request_buf, cnt,0)<0){printf("key_batch_command_Sync_error !!!!!!!\n");}
    //              printf("key_batch_end : %d\n",cnt);
//                  CpyTime[id] += _timestamp() - temp_time;
                    
                    for(j=0;j<cnt;j++){
                        volatile CPU_IPC_RW_Entry *complete_e=&globals->cpu_ipcRWQueue->entries[request_buf[j].num];
                        complete_e->return_value=request_buf[j].size;
                        __sync_synchronize();
                        (push_list[request_buf[j].num-firstSlot]) = CPU_IPC_COMPLETE;
                        __sync_synchronize();
                    }

                    cnt = 0;
                }
            }
        }
        while(f!=r){
            cudaError_t cuda_status = cudaEventQuery( events[f] );
            if(cuda_status==cudaSuccess){
                volatile CPU_IPC_RW_Entry* e = &globals->cpu_ipcRWQueue->entries[ firstSlot + event_list[f] ];
                request_buf[cnt].num=firstSlot+event_list[f];
                request_buf[cnt].offset=e->offset+(unsigned long)user_put;
                request_buf[cnt].key=e->key;
                request_buf[cnt].size=e->size;
                request_buf[cnt].mode=PUT;
                
                CpySize[id] += request_buf[cnt].size;
                
                cnt++;
                
                if (cnt >= batch_size) {
                    CpyCount[id]+=cnt;
//                  temp_time = _timestamp();
    //              printf("key_batch_start : %d\n",cnt);
                    if(key_batch_op((unsigned long)request_buf, cnt,0)<0){printf("key_batch_command_Sync_error !!!!!!!\n");}
    //              printf("key_batch_end : %d\n",cnt);
//                  CpyTime[id] += _timestamp() - temp_time;
                    
                    for(j=0;j<cnt;j++){
                        volatile CPU_IPC_RW_Entry *complete_e=&globals->cpu_ipcRWQueue->entries[request_buf[j].num];
                        complete_e->return_value=request_buf[j].size;
                        __sync_synchronize();
                        (push_list[request_buf[j].num-firstSlot]) = CPU_IPC_COMPLETE;
                        __sync_synchronize();
                    }
                    cnt = 0;
                }

                f=BUF_NEXT(f,put_num);
            }
            else break;
        }
        if (cnt != 0) {
            CpyCount[id]+=cnt;
//          temp_time = _timestamp();
        //  printf("key_batch_start : %d\n",cnt);
            if(key_batch_op((unsigned long)request_buf, cnt,0)<0){printf("key_batch_command_Sync_error !!!!!!!\n");}
    //      printf("key_batch_end : %d\n",cnt);
//          CpyTime[id] += _timestamp() - temp_time;
            
            for(j=0;j<cnt;j++){
                volatile CPU_IPC_RW_Entry *complete_e=&globals->cpu_ipcRWQueue->entries[request_buf[j].num];
                complete_e->return_value=request_buf[j].size;
                __sync_synchronize();
                (push_list[request_buf[j].num-firstSlot]) = CPU_IPC_COMPLETE;
                __sync_synchronize();
            }
            cnt = 0;
        }
    }
    for(i=0;i<put_num;i++){
        CUDA_SAFE_CALL(cudaEventDestroy(events[i]));
    }
    free(request_buf);
    free(events);
    free(event_list);
    return NULL;
}

void* rw_task_batch(void* param) //async
{
    
    TaskData* taskData = (TaskData*)param;

    int id = taskData->id;
    volatile GPUGlobals* globals = taskData->gpuGlobals;
    int num=RW_SLOTS_PER_WORKER(worker);

    int firstSlot = id * num;
    //int i;

    volatile char* push_list;
    
    //double temp_time;

    char* user_put=(globals->user_put);
    char* cuda_put=(globals->cuda_put);

    push_list=&(globals->cpu_ipcRWQueue->push_list[firstSlot]);
    
    User_list* request_buf;
    request_buf=(User_list*)malloc(sizeof(User_list)*batch_size);
/*  
    int *event_list;
    cudaEvent_t *events;
    events=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*num);
    event_list=(int*)malloc(sizeof(int)*num);
    for(i=0;i<num;i++){
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&events[i],cudaEventBlockingSync|cudaEventDisableTiming));
    }
*/
    cudaStream_t stream=globals->streamMgr->memStream[id];

    int f=0,r=0; //for event_circular queue
    int previous_f=0,previous_r=0;
    
    int cnt = 0,time_request=0;
    unsigned long long polling_start, polling_end;
    unsigned long long req_start, req_end;

    //printf("[DEBUG:host_loop.h] async batch mode start!\n");
    while (!done)
    {
        polling_start = hl_rdtsc();
        time_request=0;

        if(id==0){//first thread
            volatile cudaError_t cuda_status = cudaStreamQuery( globals->streamMgr->kernelStream );
            if ( cudaErrorNotReady != cuda_status )
            {
                pthread_mutex_lock(&task_lock);
                done = 1;
                pthread_cond_signal(&task_cond);
                pthread_mutex_unlock(&task_lock);
                continue;
            }
        }

        
        for (int i = 0; i < num; i++)
        {
            if ((push_list[i]) == CPU_IPC_PENDING) {
                req_start = hl_rdtsc();
                //  PRINT_TRACE("Handle request in worker: %d, scratch id: %d, scratch offset: %ld, request id: %d, file offset: %ld\n",id, scratchIdx, scratchSize, firstSlot + i, e->file_offset);
                (push_list[i]) = CPU_IPC_IN_PROCESS;

                time_request++;

                volatile CPU_IPC_RW_Entry* entry = &globals->cpu_ipcRWQueue->entries[firstSlot + i];
                if(entry->mode==GET){
                    //printf("[DEBUG:host_loop.h] request received!\n");
                    request_buf[cnt].offset = entry->offset;
                }
                else{
                    // PUT
                    unsigned long offset;
                    unsigned int size;

                    offset=entry->offset;
                    size=entry->size;
                    // Wait until cudaMemcpy finished
                    CUDA_SAFE_CALL(cudaMemcpyAsync(user_put+offset,cuda_put+offset,size,cudaMemcpyDeviceToHost,stream));
                    CUDA_SAFE_CALL(cudaEventRecord(put_list.events[r],stream));
                    put_list.event_list[r]=i;
                    r=BUF_NEXT(r,put_list.size);
                    CpySize[id] += request_buf[cnt].size;
                    CpyCount[id]++;

                    continue;
//                  request_buf[cnt].offset=e->offset + (unsigned long)user_put;
                }
                request_buf[cnt].key = entry->key;
                request_buf[cnt].size = entry->size;
                request_buf[cnt].mode = GET;
                request_buf[cnt].num = firstSlot + i;
                request_buf[cnt].result_offset = (unsigned long)&(entry->return_value);
                request_buf[cnt].complete_offset = (unsigned long)&(push_list[i]);
                
                CpySize[id] += request_buf[cnt].size;
                
                cnt++;
                //////////////////////

                if (cnt >= batch_size) {
                    CpyCount[id]+=cnt;
//                  temp_time = _timestamp();
//                  int tt;
//                  tt=push_list[request_buf[0].num-firstSlot];
//                  printf("before complete %d\n",cnt);
                    
                    //printf("[DEBUG:host_loop.h] send request\n");
                    if(key_batch_op((unsigned long)request_buf, cnt,1)<0){printf("key_batch_command_async_error !!!!!!!\n");}
                    req_end = hl_rdtsc();
                    clock_req += req_end - req_start;
                    req_start = req_end;
//                  while(1){
//                      if(tt!=push_list[request_buf[0].num-firstSlot]) break;
//                  }
//                  printf("after complete %d\n",cnt);
                    
//                  CpyTime[id] += _timestamp() - temp_time;
                    cnt = 0;
                }
            }
        }
        if(previous_r!=r){
//          printf("put___work!\n");
            pthread_mutex_lock(&task_lock);
            put_list.r=r; f=put_list.f;
            pthread_cond_signal(&task_cond);
            pthread_mutex_unlock(&task_lock);
            previous_r=r; previous_f=f;
        }
        // If doesn't fit on batch size
        if (cnt != 0) {
            CpyCount[id]+=cnt;
//          temp_time = _timestamp();
//          printf("before complete %d\n",cnt);
            //printf("[DEBUG:host_loop.h] send request\n");
            if(key_batch_op((unsigned long)request_buf, cnt,1)<0){printf("key_batch_command_async_error !!!!!!!\n");}
//          printf("after complete %d\n",cnt);
//          CpyTime[id] += _timestamp() - temp_time;
            cnt = 0;
            req_end = hl_rdtsc();
            clock_req += req_end - req_start;
        }
        if(time_request==0){////request 
            polling_end = hl_rdtsc();
            clock_polling += polling_end - polling_start;
            //usleep(1000*5);
        }
    }
//  for(i=0;i<num;i++){
//      CUDA_SAFE_CALL(cudaEventDestroy(events[i]));
//  }
    free(request_buf);
//  free(events);
//  free(event_list);
    return NULL;
}

void* put_async(void* param){

    TaskData* taskData = (TaskData*)param;

    int id = taskData->id;
    volatile GPUGlobals* globals = taskData->gpuGlobals;
    int num=RW_SLOTS_PER_WORKER(worker);

    int firstSlot = id * num;
    //int i;

    volatile char* push_list;

    //double temp_time;

    char* user_put=(globals->user_put);
    char* cuda_put=(globals->cuda_put);

    push_list=&(globals->cpu_ipcRWQueue->push_list[firstSlot]);

    User_list* request_buf;
    request_buf=(User_list*)malloc(sizeof(User_list)*batch_size);
    int f=0,r=0;
    int end=0,cnt;
    while(1){
//      if(done){end=1;}
        
//      printf("put_start!!!!!!\n");
        pthread_mutex_lock(&task_lock);
        put_list.f=f; r=put_list.r;
        end=done;
        if(f==r && end==0){
            pthread_cond_wait(&task_cond,&task_lock);
            put_list.f=f; r=put_list.r;
            end=done;
        }
        pthread_mutex_unlock(&task_lock);
//      printf("put f=%d, r=%d\n",f,r);
        cnt=0;
        while(f!=r){
            CUDA_SAFE_CALL(cudaEventSynchronize(put_list.events[f]));
            //          cudaError_t cuda_status = cudaEventQuery( events[f] );
            //          if(cuda_status==cudaSuccess){
            volatile CPU_IPC_RW_Entry* e = &globals->cpu_ipcRWQueue->entries[ firstSlot + put_list.event_list[f] ];
            request_buf[cnt].num=firstSlot+ put_list.event_list[f];
            request_buf[cnt].offset=e->offset+(unsigned long)user_put;
            request_buf[cnt].key=e->key;
            request_buf[cnt].size=e->size;
            request_buf[cnt].mode=PUT;
            request_buf[cnt].result_offset=(unsigned long)&(e->return_value);
            request_buf[cnt].complete_offset=(unsigned long)&(push_list[ put_list.event_list[f] ]);

            cnt++;
            if (cnt >= batch_size) {
//              printf("before put_async %d\n",cnt);
                if(key_batch_op((unsigned long)request_buf, cnt,1)<0){printf("key_batch_command_async_error !!!!!!!\n");}
//              printf("after put_async %d\n",cnt);
                cnt = 0;
            }
            f=BUF_NEXT(f,put_list.size);
        }
        if (cnt !=0) {
//          printf("before put_async %d\n",cnt);
            if(key_batch_op((unsigned long)request_buf, cnt,1)<0){printf("key_batch_command_async_error !!!!!!!\n");}
//          printf("after put_async %d\n",cnt);
            cnt = 0;
        }
        if(end==1) break;
    }
    free(request_buf);
    return NULL;
}
void run_gpufs_handler(volatile GPUGlobals* gpuGlobals, int gpuid,int pin_fd, int num_thread=1,int handle=0,int batch=128,int call=0, int run_complete=1)
{
    done = 0;
    worker=num_thread;
    batch_size=batch;

    for( int i = 0; i < worker; ++i )
    {
        CpyTime[i] = 0;
        CpyCount[i] = 0;
        CpySize[i] = 0;
    }

    double totalTime = 0,temp_time=0;

    temp_time= _timestamp();

    pthread_attr_t attr;
    pthread_attr_init( &attr );
    pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );

    if(handle==1){
        rwLoopTasksData[worker].id = pin_fd;
        rwLoopTasksData[worker].gpuGlobals =  gpuGlobals;
        rwLoopTasksData[worker].gpuid = 0;

        rwLoopTasksData[worker+1].id=0;
        rwLoopTasksData[worker+1].gpuGlobals=gpuGlobals;
        rwLoopTasksData[worker+1].gpuid=0;
        
        int num=RW_SLOTS_PER_WORKER(worker)+1;
        put_list.size=num;
        put_list.events=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*num);
        put_list.event_list=(int*)malloc(sizeof(int)*num);
        for(int i=0;i<num;i++){
            CUDA_SAFE_CALL(cudaEventCreateWithFlags(&put_list.events[i],cudaEventBlockingSync|cudaEventDisableTiming));
        }
        put_list.f=put_list.r=0;
        pthread_cond_init(&task_cond,0);
        pthread_mutex_init(&task_lock,0);

        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setdetachstate(&at,PTHREAD_CREATE_DETACHED);
        pthread_create( (pthread_t*)&(rwLoopTasksIDs[worker]), &at, complete_async, (TaskData*)&(rwLoopTasksData[worker]) ); //for complete
        
        pthread_create( (pthread_t*)&(rwLoopTasksIDs[worker+1]), &attr, put_async, (TaskData*)&(rwLoopTasksData[worker+1]) ); //for put
        
        pthread_attr_destroy(&at);
        //printf("[DEUBG:host_loop.h] complete & put thread create complete!\n");
    }
    for( int i = 0; i < worker; ++i )
    {
        rwLoopTasksData[i].id = i;
        rwLoopTasksData[i].gpuGlobals =  gpuGlobals;
        rwLoopTasksData[i].gpuid = 0;

        //pthread_spin_init(&(event_list[i].spinlock),0);
        if(handle==0){//one
            pthread_create( (pthread_t*)&(rwLoopTasksIDs[i]), &attr, rw_task, (TaskData*)&(rwLoopTasksData[i]) );
        }
        else if(handle==1){ //batch async
            pthread_create( (pthread_t*)&(rwLoopTasksIDs[i]), &attr, rw_task_batch, (TaskData*)&(rwLoopTasksData[i]) );
        }
        else{ //batch sync
            pthread_create( (pthread_t*)&(rwLoopTasksIDs[i]), &attr, rw_task_batch_sync, (TaskData*)&(rwLoopTasksData[i]) );
        }
    }
    //printf("[DEUBG:host_loop.h] get thread create complete!\n");

    pthread_attr_destroy( &attr );

//  mainLoop( gpuGlobals, gpuid );

    for( int i = 0; i < worker; ++i )
    {
        pthread_join( rwLoopTasksIDs[i], NULL );
    }
    // complete thread join???
    if(handle==1){
        // put thread join
        pthread_join(rwLoopTasksIDs[worker+1],NULL);
        int ret=ioctl(pin_fd,I_KEY_ASYNC_CLOSE,(unsigned long)0);
        if(ret!=0) printf("ASYNC_CLOSE error!\n");
    }
    double end_time=_timestamp();
    totalTime = elapsed_time(temp_time,end_time);
    
    if(handle==1){
        free(put_list.event_list);
        for(int i=0;i<RW_SLOTS_PER_WORKER(worker)+1;i++){
            CUDA_SAFE_CALL(cudaEventDestroy(put_list.events[i]));
        }
        free(put_list.events);
    }

    int totalCount = 0;
    unsigned long totalSize = 0;

    for (int i = 0; i < worker; i++)
    {
        PRINT_TIMES( "memcpy [%d]:\n", i );
//      PRINT_TIMES( "\tTime: %lfms\n", CpyTime[i] / 1e3);
        PRINT_TIMES( "\tCount: %d\n", CpyCount[i]);
        PRINT_TIMES( "\tSize: %lfMB\n", ((double)CpySize[i] / (1<<20)) );
        if( CpyCount[i] > 0 )
        {
            PRINT_TIMES( "\tAverage buffer size: %luKB\n", (CpySize[i] >> 10) / CpyCount[i]);
        }
        else
        {
            PRINT_TIMES( "\tAverage buffer size: 0KB\n");
        }
//      PRINT_TIMES( "\tBandwidth: %fGB/s\n\n", ((double)CpySize[i] / (1 << 30)) / (CpyTime[i] / 1e6));

        totalCount += CpyCount[i];
        totalSize += CpySize[i];
    }

    printf("GPUKV | Total time: %lf s\tCount: %d\tSize: %lf MB, Bandwidth: %lf GB/s, clock used for polling: %lld\n", totalTime,totalCount,(double)totalSize/(1<<20),((double)totalSize/(1<<30))/totalTime, clock_polling    );////1e6);  //ms....
//  printf( "\tBandwidth: %lfGB/s\n\n", ((double)totalSize / (1 << 30)) / (totalTime / 1e6));//1e9));
//  PRINT_TIMES( "\tCount: %d\n", totalCount);
//  PRINT_TIMES( "\tSize: %lf MB\n",((double)totalSize/(1<< 20) ) );
    if( totalCount > 0 )
    {
        PRINT_TIMES( "\tAverage buffer size: %luKB\n", (totalSize >> 10) / totalCount);
    }
    else
    {
        PRINT_TIMES( "\tAverage buffer size: 0KB\n");
    }
    //GPU STAT PRINT!
    PRINT_TIMES("\nGPU_STATS\n");
    
    int num = 0;
    if(call==0) num=totalCount/32; //thread 
    else num=totalCount;
    
#ifdef TRACE
    //printf("GPUKV | GPUSTATS count : %d\n",num);
//  PRINT_TIMES("count for time : %d\n",num);
    
//  PRINT_STATS(numRequest);
    PRINT_TIME(CPUReadTime,num);
#endif
    PRINT_TIME(LatencyTime,num);
//  PRINT_TIMES("\n\n");
    printf("clock for polling: %lld\n", clock_polling);
    printf("clock for request: %lld\n", clock_req);
}

#endif
