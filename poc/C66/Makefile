# Builds test.cu with CUDA 13.3 ptxas -O0/-O1/-O2 and prints each checksum.
NVCC ?= /export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.3/bin/nvcc
FLAGS = -std=c++17 -arch=sm_86 -w -ccbin /usr/bin/g++-10
SOURCE = min.cu
all:
	$(NVCC) $(FLAGS) -Xptxas=-O0 $(SOURCE) -o O0
	$(NVCC) $(FLAGS) -Xptxas=-O1 $(SOURCE) -o O1
	$(NVCC) $(FLAGS) -Xptxas=-O2 $(SOURCE) -o O2
	@printf "ptxas-O0 checksum=%s\n" "$$(./O0 | sed -n 's/^checksum=//p')"
	@printf "ptxas-O1 checksum=%s\n" "$$(./O1 | sed -n 's/^checksum=//p')"
	@printf "ptxas-O2 checksum=%s\n" "$$(./O2 | sed -n 's/^checksum=//p')"

# Minimal core reproducer (see README section "Minimal core (min.cu)").
MIN_SOURCE = min.cu
min:
	$(NVCC) $(FLAGS) -Xptxas=-O0 $(MIN_SOURCE) -o min_O0
	$(NVCC) $(FLAGS) -Xptxas=-O1 $(MIN_SOURCE) -o min_O1
	@printf "min ptxas-O0 checksum=%s (expect 16991622330266937562)\n" "$$(./min_O0 | sed -n 's/^checksum=//p')"
	@printf "min ptxas-O1 checksum=%s (expect 15140686756037123777)\n" "$$(./min_O1 | sed -n 's/^checksum=//p')"

clean:
	rm -f O0 O1 O2 min_O0 min_O1
