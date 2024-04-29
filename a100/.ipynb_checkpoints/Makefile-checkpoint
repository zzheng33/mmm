# Makefile to compile CUDA programs

# Compiler
NVCC = nvcc

# Targets
all: v1 v2

v1: mmm-a100-version1.cu
	$(NVCC) mmm-a100-version1.cu -o v1

v2: mmm-a100-version2.cu
	$(NVCC) mmm-a100-version2.cu -o v2

clean:
	rm -f v1 v2
