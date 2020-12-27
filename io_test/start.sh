#!/bin/bash

echo "[DEBUG-shell] start benchmark.py"
echo "[DEBUG] drop caches"

./reset_cache

##### Data feeding ######
#./bin/utils/primitive_put -s 4096 -n 4194304

#./bin/utils/graph_put 1971281 ./dataset/roadnetCA/
#./bin/utils/graph_put 4194304 ./dataset/synthetic/

#nvprof --log-file ./log/gpukv.%p.txt --export-profile ./log/gpukv.%p.nvvp --print-gpu-trace --profile-child-processes --profile-api-trace driver --source-level-analysis global_access --source-level-analysis shared_access --source-level-analysis instruction_execution --trace gpu ./bin/synthetic/gpukv -s 4096 -n 4194304 -m 2 -p 1 -i 1 -b 256 -t 128 -c 1

function run_iter(){
    local core=$1
    local hz=$2
    local mode=$3
    local app=$4
    local option=$5
    local -n thread_list=$6

    ./set_cpu $core $hz
    echo -e "\n`date`\n/// $core Core $hz" >> ./result/$mode\_$app.txt
    for t in ${thread_list[@]};do
        python3 bench.py $mode $app $option --cpu_thread=$t
    done
}

#threads=(256 128 64 32 16 8 4) 
#threads=(128 64 32 16 8 4) 
#threads=(256 4)
max_thread=(256)
min_thread=(4)
gpukv_threads=(1)

##### BFS #####
#run_iter 4 3.5GHz rocks_bfs graph '--size=4 --unit=thread --handle=async_batch' threads
#run_iter 4 800MHz rocks_bfs graph '--size=4 --unit=thread --handle=async_batch' threads
#run_iter 2 800MHz rocks_bfs graph '--size=4 --unit=thread --handle=async_batch' threads
#run_iter 4 3.5GHz rocks_bfs_direct graph '--size=4 --unit=thread --handle=async_batch' threads
#run_iter 2 800MHz rocks_bfs_direct graph '--size=4 --unit=thread --handle=async_batch --repeat=3' threads

#run_iter 4 3.5GHz GPUKV_bfs graph '--size=4 --unit=thread --handle=async_batch' gpukv_threads 
#run_iter 4 3.5GHz conventional_bfs graph '--size=4 --unit=thread --handle=async_batch' threads

#./set_cpu 4 800MHz
#run_iter 2 800MHz GPUKV_bfs graph '--size=4 --unit=thread --handle=async_batch' gpukv_threads
#run_iter 2 800MHz conventional_bfs graph '--size=4 --unit=thread --handle=async_batch' threads


##### Page Rank #####

#run_iter 4 3.5GHz rocks_pr graph '--size=4 --unit=thread --handle=async_batch' threads
#run_iter 4 3.5GHz rocks_pr_direct graph '--size=4 --unit=thread --handle=async_batch' threads

#./set_cpu 4 800MHz
#run_iter 4 3.5GHz rocks_pr graph '--size=4 --unit=thread --handle=async_batch' threads
#run_iter 4 3.5GHz rocks_pr_direct graph '--size=4 --unit=thread --handle=async_batch' threads

#run_iter 4 3.5GHz GPUKV_pr graph '--size=4 --unit=thread --handle=async_batch' gpukv_threads 
#./set_cpu 4 800MHz
#run_iter 2 800MHz GPUKV_pr graph '--size=4 --unit=thread --handle=async_batch' gpukv_threads
#run_iter 4 3.5GHz conventional_pr graph '--size=4 --unit=thread --handle=async_batch' threads
#run_iter 2 800MHz conventional_pr graph '--size=4 --unit=thread --handle=async_batch' threads

##### Synthetic Workloads #####

####### KVSSD GPUKV / conventional setting ########
#run_iter 4 3.5GHz kvssd streaming '--size=4 --requests=4194304 --gpu_thread=32 --gpu_block=32' max_thread
#run_iter 4 3.5GHz gpukv streaming '--size=4 --handle=async_batch --requests=4194304' gpukv_threads 
#run_iter 4 3.5GHz kvssd dynamic '--size=4 --requests=4194304 --gpu_thread=32 --gpu_block=32' max_thread
#run_iter 4 3.5GHz gpukv dynamic '--size=4 --handle=async_batch --requests=4194304' gpukv_threads 

#./set_cpu 4 800MHz
#run_iter 2 800MHz gpukv streaming '--size=4 --handle=async_batch --requests=4194304' gpukv_threads
#run_iter 2 800MHz gpukv dynamic '--size=4 --handle=async_batch --requests=4194304' gpukv_threads
#run_iter 2 800MHz kvssd streaming '--size=4 --requests=4194304 --gpu_thread=32 --gpu_block=32' min_thread
#run_iter 2 800MHz kvssd dynamic '--size=4 --requests=4194304 --gpu_thread=32 --gpu_block=32' min_thread


####### rocksDB setting #######

#run_iter 4 3.5GHz rocksdb_default streaming '--size=4 --requests=4194304' max_thread
#run_iter 4 3.5GHz rocksdb_default dynamic   '--size=4 --requests=4194304' max_thread
#run_iter 4 3.5GHz rocksdb_direct streaming  '--size=4 --requests=4194304' max_thread
#run_iter 4 3.5GHz rocksdb_direct dynamic    '--size=4 --requests=4194304' max_thread

#./set_cpu 4 800MHz
#run_iter 4 800MHz rocksdb_default streaming '--size=4 --requests=4194304' min_thread
#run_iter 2 800MHz rocksdb_default dynamic   '--size=4 --requests=4194304' min_thread
#run_iter 2 800MHz rocksdb_direct streaming  '--size=4 --requests=4194304' min_thread
#run_iter 2 800MHz rocksdb_direct dynamic    '--size=4 --requests=4194304' min_thread

