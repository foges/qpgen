
NVCC=nvcc
NVFLAGS=-arch=sm_20
LDFLAGS=-L/usr/local/cuda/lib -lcublas -lcusparse -lcudart

qpgen_ln.o: qpgen.o
	$(NVCC) $(NVFLAGS) $< -dlink -o $@

qpgen.o: qpgen.cu qpgen.h linalg.h cu_linalg.h mattypes.h
	$(NVCC) $(NVFLAGS) $< -dc -o $@

clean:
	rm *.o main

