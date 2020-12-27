t?=1
l?=0
d?=16

.PHONY: all microbench utils graph synthetic

all: microbench utils graph synthetic


microbench:
	cd microbench; make t=$(t) l=$(l) d=$(d); cd ..

synthetic:
	cd synthetic; make t=$(t) l=$(l) d=$(d); cd ..

utils:
	cd utils; make t=$(t) l=$(l) d=$(d); cd ..

bfs:
	cd graph_workload/$@; make t=$(t) l=$(l) d=$(d); cd ../..

pr:
	cd graph_workload/$@; make t=$(t) l=$(l) d=$(d); cd ../..

graph: bfs pr

clean:
	rm -rf bin/
	cd graph_workload/bfs; make clean; cd ../..
	cd graph_workload/pr; make clean; cd ../..
	cd microbench; make clean; cd ..
	cd utils; make clean; cd ..
	cd synthetic; make clean; cd ..
