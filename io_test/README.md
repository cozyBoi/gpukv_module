select / mode / size / call / handle / thread / --r / --f
* select
    * 0: original(pipeline)
    * 1: GPUKV (cp)
    * 2: original2 (irregular)
    * 3: simple(non-pipeline)
    * 4: max (CPU\_COPY)
* mode
    * 0: GET
    * 1: PUT
    * 2: MODIFY
    * 3: APP(sobel)
	* 4: MODIFY ver.2
* size - request size
    * \# of KB
* call
    * 0: thread
	* 1: warp
	* 2: thread block
* handle
    * 0: one 
	* 1: batch\_async
	* 2: batch\_sync
* thread - CPU side host thread (num of worker)
* --r: repeat
* --f: do first.cu
* --n: number of handled request per worker
* MODIFY2: dynamic(irregular) workload.
    - irregular workload는 예측할 수 없음
	- gpu kernel의 메인 thread가 하나임
	- 1 thread로 call

```
GPUKV / MODIFY / 768 / thread_block / batch_async / 1 thread
original / MODIFY / 768 / thread / / 4 thread
original / MODIFY / 768 / thread / / 1 thread

GPUKV / MODIFY2 / 768 / thread_block / batch_async / 1 thread
original2 / MODIFY2 / 768 / thread / / 1 thread
SIMPLE / MODIFY2 / 768 / thread / / 1 thread

SIMPLE / GET / 768 / thread / / 4 thread
original / GET / 768 / thread / / 4 thread
GPUKV / GET / 768 / thread_block / batch_async / 1 thread
max / GET / 768 / thread / / 8 thread

GPUKV / GET / 768 / thread_block / one / 1 thread
GPUKV / GET / 768 / thread_block / one / 4 thread
GPUKV / GET / 768 / thread_block / one / 8 thread

GPUKV / GET / 768 / thread_block / batch_sync / 1 thread
GPUKV / GET / 768 / thread_block / batch_async / 1 thread

GPUKV / GET / 768 / thread_block / one / 1 t / block=64
GPUKV / GET / 768 / thread_block / one / 4 t / block=64
GPUKV / GET / 768 / thread_block / one / 8 t / block=64
GPUKV / GET / 768 / thread_block / batch_sync / 1 t / block=64
GPUKV / GET / 768 / thread_block / batch_async / 1 t / block=64

GPUKV / GET / 768 / thread_block / one / 1 t / block=16 / batch=16
GPUKV / GET / 768 / thread_block / one / 4 t / block=16 / batch=16
GPUKV / GET / 768 / thread_block / one / 8 t / block=16 / batch=16
GPUKV / GET / 768 / thread_block / batch_sync / 1 t / block=16 / batch=16
GPUKV / GET / 768 / thread_block / batch_async / 1 t / block=16 / batch=16
```
