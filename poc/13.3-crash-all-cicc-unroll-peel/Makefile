NVCC ?= nvcc
ARCH ?= sm_75
OPT ?= 0
NVCCFLAGS ?= -std=c++17

SRC := test.cu
BUILD_DIR ?= build
PTX := $(BUILD_DIR)/test.ptx

override CUDA_VERSION_MIN := 12.5
override CUDA_VERSION_MAX := 13.3

.DEFAULT_GOAL := all
.PHONY: all build reproduce version version-assert clean

all: build

# Compile the reduced source to PTX.  The expected result for a compiler in
# the accepted range is a cicc SIGSEGV while processing this translation unit.
build: $(PTX)

reproduce: build

$(PTX): $(SRC) | version-assert
	@mkdir -p "$(@D)"
	@printf "BUILD %s (nvcc -O%s -arch=%s)\n" "$<" "$(OPT)" "$(ARCH)"
	@$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -O$(OPT) -ptx "$<" -o "$@"

# Keep this check in the Makefile so the sample cannot silently be run with a
# toolchain outside the range in which the crash was verified.
version-assert:
	@set -eu; \
	assert() { \
		test "$$1" = 1 || { echo "ASSERTION FAILED: $$2" >&2; exit 1; }; \
	}; \
	version="$$( $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1.\2/p' | head -n 1 )"; \
	assert "$$([ -n "$$version" ] && echo 1 || echo 0)" \
		"could not determine the nvcc version from '$(NVCC) --version'"; \
	code="$$(printf '%s\n' "$$version" | awk -F. '{print $$1 * 100 + $$2}')"; \
	min="$$(printf '%s\n' "$(CUDA_VERSION_MIN)" | awk -F. '{print $$1 * 100 + $$2}')"; \
	max="$$(printf '%s\n' "$(CUDA_VERSION_MAX)" | awk -F. '{print $$1 * 100 + $$2}')"; \
	assert "$$([ "$$code" -ge "$$min" ] && [ "$$code" -le "$$max" ] && echo 1 || echo 0)" \
		"nvcc $$version is outside the supported range $(CUDA_VERSION_MIN)-$(CUDA_VERSION_MAX)"; \
	echo "nvcc $$version (supported: $(CUDA_VERSION_MIN)-$(CUDA_VERSION_MAX))"

version:
	@$(NVCC) --version

clean:
	$(RM) -r "$(BUILD_DIR)"
