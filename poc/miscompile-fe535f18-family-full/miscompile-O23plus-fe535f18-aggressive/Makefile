NVCC ?= /usr/local/cuda-13.3/bin/nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -w
TIMEOUT ?= 10
SRC ?= reduced.cu
EXPECTED ?= 4096945828
OLEVELS := 0 1 2 3
BUILD_DIR := build
BIN := repro

BINS := $(foreach opt,$(OLEVELS),$(BUILD_DIR)/O$(opt)/$(BIN))
RUN_TARGETS := $(addprefix run-O,$(OLEVELS))

.DEFAULT_GOAL := run

.PHONY: all build run clean version

all: run

build: $(BINS)

$(BUILD_DIR)/O%/$(BIN): $(SRC)
	@mkdir -p $(@D)
	@printf "BUILD O%s\n" "$*"
	@$(NVCC) -std=c++17 -arch=$(ARCH) $(NVCCFLAGS) -Xptxas=-O$* $< -o $@

run: build
	@printf "nvcc: "
	@$(NVCC) --version | tail -n 1
	@printf "arch: %s\n" "$(ARCH)"
	@printf "%-4s %-8s %s\n" "opt" "status" "checksum"
	@$(MAKE) --no-print-directory $(RUN_TARGETS)

run-O%: $(BUILD_DIR)/O%/$(BIN)
	@out="$$(timeout $(TIMEOUT) ./$<)"; \
	checksum="$$(printf '%s\n' "$$out" | sed -n 's/^checksum=//p' | tail -n 1)"; \
	if [ "$$checksum" = "$(EXPECTED)" ]; then status=CORRECT; else status=WRONG; fi; \
	printf "%-4s %-8s %s\n" "O$*" "$$status" "$$checksum"

version:
	$(NVCC) --version

clean:
	$(RM) -r $(BUILD_DIR)
