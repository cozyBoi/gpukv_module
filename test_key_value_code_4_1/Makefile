PREFIX=$(PWD)
CUDA=/usr/local/cuda-10.1

CUDA_LIB := -L $(CUDA)/lib64 -L $(CUDA)/lib -L /usr/lib64/nvidia -L /usr/lib/nvidia
CUDA_INC += -I $(CUDA)/include

CPPFLAGS := $(CUDA_INC) -I $(PWD)/kernel_code/module/
LDFLAGS := $(CUDA_LIB) -L $(CUDA)/lib64
COMMONFLAGS := -O2
CFLAGS   += $(COMMONCFLAGS)
CXXFLAGS += $(COMMONCFLAGS)
LIBS     := -lcudart -lcuda -lpthread -ldl

SRCS := validate.cpp copybw.cpp normal.cpp
EXES := $(SRCS:.cpp=)

all: exes

exes: $(EXES)


validate.o: validate.cpp 
copybw.o: copybw.cpp
normal.o: normal.cpp

validate: validate.o 
	$(LINK.cc)  -o $@ $^ $(LIBS)

copybw: copybw.o 
	$(LINK.cc)  -o $@ $^ $(LIBS)
normal: normal.o 
	$(LINK.cc)  -o $@ $^ $(LIBS)


clean:
	rm -f *.o $(EXES) lib*.{a,so}* *~ core.*
