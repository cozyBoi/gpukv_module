#!/bin/bash

sync; echo 1 > /proc/sys/vm/drop_caches
sync; echo 2 > /proc/sys/vm/drop_caches
sync; echo 3 > /proc/sys/vm/drop_caches

#./get_bd_original > breakdown_cache_original.txt
./get_bd_original > breakdown_direct_original.txt
