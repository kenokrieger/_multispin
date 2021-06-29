CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
CC=gcc
LD=$(CUDACC)
CFLAGS=-c -O3 -g -I$(CUDA_HOME)/include
CUDACFLAGS=-c -O3 --use_fast_math -lineinfo -arch=sm_61 -Xptxas=-v
LDFLAGS= -Xcompiler=-fopenmp

all: build/cuIsing

build/cuIsing: build/main.o build/utils.o
	$(LD) -o build/cuIsing build/main.o build/utils.o $(LDFLAGS)

build/%.o: src/%.cu
	$(CUDACC) $(CUDACFLAGS) $< -o $@

build/%.o: src/%.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm build/*.o build/cuIsing
