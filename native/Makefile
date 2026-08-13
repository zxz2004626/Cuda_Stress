# Builds cuda_runner against the CUDA Driver API only.
#
# cuda.h and libcuda.so come from the toolkit / driver.  We deliberately do NOT link
# the CUDA runtime, so the binary is portable across all installed toolkits and only
# depends on the system driver (libcuda).
#
# CUDA_ROOT defaults to a pre-13 toolkit: CUDA 13 moved cuCtxCreate to a new
# CUctxCreateParams signature, but the runner only needs the stable Driver API surface
# exposed by e.g. the 12.6 header.  Link still goes against the system driver libcuda.

CUDA_ROOT ?= /usr/local/cuda-11.7
CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++14 -Wall -Wextra -I$(CUDA_ROOT)/include
LDFLAGS   ?= -L/usr/lib/x86_64-linux-gnu -lcuda -lpthread

cuda_runner: cuda_runner.cpp
	$(CXX) $(CXXFLAGS) cuda_runner.cpp $(LDFLAGS) -o $@

clean:
	rm -f cuda_runner

.PHONY: clean
