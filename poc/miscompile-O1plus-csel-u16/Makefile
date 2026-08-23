NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -w
TIMEOUT ?= 10
SRC := test.cu
BUILD_DIR := build
BIN := csel_u16
EXPECTED ?= 9115
OLEVELS := 0 1 2 3

# Only allow nvcc versions in the known-trigger range (major.minor).
NVCC_VERSION_MIN ?= 13.1
NVCC_VERSION_MAX ?= 13.3

BINS := $(foreach opt,$(OLEVELS),$(BUILD_DIR)/O$(opt)/$(BIN))
RUN_TARGETS := $(addprefix run-O,$(OLEVELS))

.DEFAULT_GOAL := run

.PHONY: all build run clean version version-assert

all: run

build: version-assert $(BINS)

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

version-assert:
	@v=$$($(NVCC) --version | sed -n 's/^Build cuda_\([0-9]*\)\.\([0-9]*\)\..*/\1.\2/p'); \
	if [ -z "$$v" ]; then \
	  echo "ERROR: could not parse nvcc version from '$(NVCC) --version'" >&2; exit 1; \
	fi; \
	cur=$$(echo "$$v" | awk -F. '{print $$1*100+$$2}'); \
	min=$$(echo "$(NVCC_VERSION_MIN)" | awk -F. '{print $$1*100+$$2}'); \
	max=$$(echo "$(NVCC_VERSION_MAX)" | awk -F. '{print $$1*100+$$2}'); \
	echo "nvcc version: $$v"; \
	if [ "$$cur" -lt "$$min" ] || [ "$$cur" -gt "$$max" ]; then \
	  echo "ERROR: nvcc $$v is outside the expected range $(NVCC_VERSION_MIN) - $(NVCC_VERSION_MAX) for this reproducer." >&2; exit 1; \
	else \
	  echo "OK: nvcc $$v is in the expected range $(NVCC_VERSION_MIN) - $(NVCC_VERSION_MAX)."; \
	fi

clean:
	$(RM) -r $(BUILD_DIR)
