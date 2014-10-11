CXX=g++
CXXFLAGS=-Wall
LDFLAGS=-L/usr/local/cuda/lib -lcublas -lcusparse -lcudart

NVCC=nvcc
NVFLAGS=-arch=sm_20

all: main.cpp qpgen.o qpgen_ln.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o main

qpgen_ln.o: qpgen.o
	$(NVCC) $(NVFLAGS) $< -dlink -o $@

qpgen.o: qpgen.cu qpgen.h linalg.h cu_linalg.h mattypes.h
	$(NVCC) $(NVFLAGS) $< -dc -o $@

clean:
	rm *.o main

