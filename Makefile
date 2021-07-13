CUDA_HOME = /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

CUDACFLAGS=-c -O3 --use_fast_math -lineinfo -arch=sm_61 -Xptxas=-v
LDFLAGS= -Xcompiler=-fopenmp
OBJECT_FILES = build/main.o

all: build/cuIsing

build/cuIsing: $(OBJECT_FILES)
	$(NVCC) -o build/cuIsing $(OBJECT_FILES) $(LDFLAGS)

build/%.o: src/%.cu
	$(NVCC) $(CUDACFLAGS) $< -o $@

clean:
	rm build/*.o build/cuIsing
