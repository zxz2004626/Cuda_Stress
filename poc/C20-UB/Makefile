# Builds test.cu with CUDA 13.3 ptxas -O0/-O1/-O2 and prints each checksum.
NVCC_13.3 ?= /export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.3/bin/nvcc
NVCC_11.1 ?= /export/ssd1/xzhanglb/nvcc-toolkit/cuda-11.1/bin/nvcc
FLAGS = -std=c++17 -arch=sm_86 -w -ccbin /usr/bin/g++-10

all:
	$(NVCC_11.1) $(FLAGS) -Xptxas=-O1 test.cu -o 11O1
	$(NVCC_13.3) $(FLAGS) -Xptxas=-O0 test.cu -o 13O0
	$(NVCC_13.3) $(FLAGS) -Xptxas=-O1 test.cu -o 13O1
	$(NVCC_13.3) $(FLAGS) -Xptxas=-O2 test.cu -o 13O2
	@printf "11.1 ptxas-O1 checksum=%s\n" "$$(./11O1 | sed -n 's/^checksum=//p')"
	@printf "13.3 ptxas-O0 checksum=%s\n" "$$(./13O0 | sed -n 's/^checksum=//p')"
	@printf "13.3 ptxas-O1 checksum=%s\n" "$$(./13O1 | sed -n 's/^checksum=//p')"
	@printf "13.3 ptxas-O2 checksum=%s\n" "$$(./13O2 | sed -n 's/^checksum=//p')"

clean:
	rm -f 11O1 13O0 13O1 13O2
