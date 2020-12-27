#!/usr/bin/python3
import argparse
import multiprocessing
import subprocess
import signal
import os
import time


#print("[DEBUG:bench.py] Argument parsing")
print()

CHOICES = {
    "mode": ["original", "GPUKV", "simple", "max", # original microbench(synthetic)
        "gpukv", "kvssd", "rocksdb_direct", "rocksdb_default", # synthetic
        "GPUKV_bfs", "conventional_bfs", "rocks_bfs", "rocks_bfs_direct", # bfs
        "GPUKV_pr", "conventional_pr", "rocks_pr", "rocks_pr_direct", # pr
        ],
    "app":  ["GET", "PUT", "MODIFY", "sobel", "MODIFY2", "graph", "streaming", "dynamic"],
    "unit": ["thread", "warp", "thread_block"],
    "handle": ["sync_1t_no_batch", "sync_nt_no_batch",
               "sync_batch", "async_batch"],
}

FIRST_NEEDED_CASE = ["GET", "sobel", "MODIFY", "MODIFY2"]

CONVERTER = {
    "mode": {
        "original": "bin/micro/pipeline",
        "GPUKV": "bin/micro/gpukv",
        "simple": "bin/micro/nonpipe",
        "max": "max",
        # synthetic workload
        "gpukv": "bin/synthetic/gpukv",
        "kvssd": "bin/synthetic/conv_kvssd",
#        "rocksdb_direct": "bin/synthetic/conv_rocks_direct",
#        "rocksdb_default": "bin/synthetic/conv_rocks_default",
        # CPP version
        "rocksdb_direct": "bin/synthetic/conv_rocks_cpp_direct",
        "rocksdb_default": "bin/synthetic/conv_rocks_cpp_default",
        # bfs
        "GPUKV_bfs": "bin/graph/bfs_gpukv",
        "conventional_bfs": "bin/graph/bfs_conv",
        "rocks_bfs": "bin/graph/bfs_conv_rocks_cpp",
        "rocks_bfs_direct": "bin/graph/bfs_conv_rocks_cpp_direct",
        # pr
        "GPUKV_pr": "bin/graph/pr_gpukv",
        "conventional_pr": "bin/graph/pr_conv",
        "rocks_pr": "bin/graph/pr_conv_rocks",
        "rocks_pr_direct": "bin/graph/pr_conv_rocks_direct",
    },
    "app": {
        "GET": 0,
        "PUT": 1,
        "MODIFY": 2,
        "sobel": 3,
        "MODIFY2": 4,
        "graph": 5,
        "streaming": 2,
        "dynamic": 4,
    },
    "unit": {
        "thread": 0,
        "warp": 1,
        "thread_block": 2,
    },
    "handle": {
        "sync_1t_no_batch": 0,
        "sync_nt_no_batch": 0,
        "sync_batch": 2,
        "async_batch": 1,
    },
}

parser = argparse.ArgumentParser(description="Running benchmark for GPUKV")
parser.add_argument('mode', type=str, default='GPUKV',
                    choices=CHOICES['mode'], help="Running type")
parser.add_argument('app', type=str, default='MODIFY',
                    choices=CHOICES['app'], help="Running application")
parser.add_argument('--requests', type=int, default=32768,
                    help="Number of requests at application")
parser.add_argument('--size', type=int, default=1024,
                    help="Size of data to run application(Unit: KB)")

parser.add_argument('--unit', type=str, default='warp',
                    choices=CHOICES['unit'],
                    help="Unit for managing GPU-side processing")
parser.add_argument('--handle', type=str, default='async_batch',
                    choices=CHOICES['handle'], help="Managing type for GPUKV")
parser.add_argument('--cpu_thread', type=int, default=0,
                    help="Number of host-side managing thread")

parser.add_argument('--repeat', type=int, default=1,
                    help="Number of repeat benchmark")
parser.add_argument('--do-first', dest='do_first', action='store_true',
                    help="In using GET case, do first or not")
parser.add_argument('--gpu_block', type=int, default=256,
                    help="Number of GPU blocks")
parser.add_argument('--gpu_thread', type=int, default=128,
                    help="Number of GPU threads per block")
parser.add_argument('--batch', type=int, default=64, help="Size of batching")
parser.set_defaults(do_first=False)

args = parser.parse_args()
args = vars(args)


request_size = args['size']

if args['app'] == 'sobel':
    request_size = 768
    request_cnt = 1024*16
else:
    """
    cnt:  8, 32, 1024*4, 1024*32, 2048,
    """
    max_cnt = 1024 * 32
    request_cnt = max_cnt

if args['requests']:
    request_cnt = args['requests']

byte_size = request_size * 1024
mb_size = float(byte_size * request_cnt) / (1024*1024)
#print("[DEBUG:bench.py] {} \n request_size: {:.2f}KB request_cnt: {} total: {:.2f}MB".format(
#args, request_size, request_cnt, mb_size))

if args['app'] in FIRST_NEEDED_CASE and args['do_first']:
    #print("[DEBUG:bench.py] Do first.cu")
    os.system(f"./first -s {byte_size} -n {request_cnt}")

def mpstat(stat):
    start = "mpstat 1 1 | awk 'FNR == 1 {print $6, $7} FNR == 5 {print \"[START] " + stat + "\", $1, 100-$12}'| tee -a stat.txt"
    run = "mpstat 1 1 | tail -1 | awk '{print \"[RUN  ] " + stat + "\", $1, 100-$12}'| tee -a stat.txt"
    os.system(start)
    while True:
        os.system(run)

stat = f"{args['mode']} | application: {args['app']:7s} size: {request_size:7.2f}KB cnt: {request_cnt:5d}"
end = "mpstat 1 1 | tail -1 | awk '{print \"[END  ] " + stat + "\", $1, 100-$12}'| tee -a stat.txt"
run = "./{0} -s {1} -n {2} -m {3} -p {4} -i {5} -b {6} -t {7} -c {8} | tee -a result/{9}_{10}.txt".format(
        CONVERTER['mode'][args['mode']],
        byte_size,
        request_cnt,
        CONVERTER['app'][args['app']],
        args['cpu_thread'],
        CONVERTER['handle'][args['handle']],
        args['gpu_block'],
        args['gpu_thread'],
        CONVERTER['unit'][args['unit']],
        args['mode'],
        args['app']
)

drop_cache = "sync; echo 1 > /proc/sys/vm/drop_caches;sync; echo 2 > /proc/sys/vm/drop_caches;sync; echo 3 > /proc/sys/vm/drop_caches"

print(f"[DEBUG:bench.py] Test start. Repeat: {args['repeat']}")
for i in range(0, args['repeat']):
    os.system(drop_cache)
    print(f"[DEBUG:bench.py] Iteration: {i+1}")
    #proc = multiprocessing.Process(target=mpstat, args=(stat, ))
    #proc.start()
    #print(f"[DEBUG:bench.py] mpstat executed")

    # For forcible terminate of perf
    """procs = []
    procs.append(subprocess.Popen("perf stat -a -e power/energy-gpu/ -I 1000 -o power/gpu.txt --append",
        shell=True, preexec_fn=os.setsid))
    procs.append(subprocess.Popen("perf stat -a -e power/energy-cores/ -I 1000 -o power/cpu.txt --append",
        shell=True, preexec_fn=os.setsid))
    procs.append(subprocess.Popen("perf stat -a -e power/energy-ram/ -I 1000 -o power/ram.txt --append",
        shell=True, preexec_fn=os.setsid))
    print(f"[DEBUG:bench.py] perf execution completed")
"""
    print(run)
    os.system(run)
    #proc.terminate()
    #os.system(end)
    """for process in procs:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    """

    if i+1 != args['repeat']:
        time.sleep(3)
