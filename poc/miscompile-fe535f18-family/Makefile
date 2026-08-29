NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -w
TIMEOUT ?= 10

VARIANT ?= o1only
SRC := $(VARIANT).cu
BUILD_DIR := build/$(VARIANT)
BIN := repro
OLEVELS := 0 1 2 3

EXPECTED_o1only := 2053990987
EXPECTED_o1plus-cv15-indexcast := 2053990987
EXPECTED_o23plus-aggressive := 1442507339
EXPECTED_o23plus-i1trip := 4123129483
EXPECTED := $(EXPECTED_$(VARIANT))

BINS := $(foreach opt,$(OLEVELS),$(BUILD_DIR)/O$(opt)/$(BIN))
RUN_TARGETS := $(addprefix run-O,$(OLEVELS))

.DEFAULT_GOAL := run

.PHONY: all build run run-all clean version variants

all: run

variants:
	@printf "%s\n" \
	  "o1only" \
	  "o1plus-cv15-indexcast" \
	  "o23plus-aggressive" \
	  "o23plus-i1trip"

build: $(BINS)

$(BUILD_DIR)/O%/$(BIN): $(SRC)
	@mkdir -p $(@D)
	@printf "BUILD %-28s O%s\n" "$(VARIANT)" "$*"
	@$(NVCC) -std=c++17 -arch=$(ARCH) $(NVCCFLAGS) -Xptxas=-O$* $< -o $@

run: build
	@if [ -z "$(EXPECTED)" ]; then \
	  printf "Unknown VARIANT=%s. Try: make variants\n" "$(VARIANT)"; \
	  exit 2; \
	fi
	@printf "nvcc: "
	@$(NVCC) --version | tail -n 1
	@printf "arch: %s\n" "$(ARCH)"
	@printf "variant: %s\n" "$(VARIANT)"
	@printf "%-4s %-8s %s\n" "opt" "status" "output"
	@$(MAKE) --no-print-directory $(RUN_TARGETS)

run-all:
	@for variant in o1only o1plus-cv15-indexcast o23plus-aggressive o23plus-i1trip; do \
	  printf "\n==== %s ====\n" "$$variant"; \
	  $(MAKE) --no-print-directory VARIANT=$$variant run; \
	done

run-O%: $(BUILD_DIR)/O%/$(BIN)
	@out="$$(timeout $(TIMEOUT) ./$<)"; \
	value="$$(printf '%s\n' "$$out" | sed -n 's/^output=//p' | tail -n 1)"; \
	if [ "$$value" = "$(EXPECTED)" ]; then status=CORRECT; else status=WRONG; fi; \
	printf "%-4s %-8s %s\n" "O$*" "$$status" "$$value"

version:
	$(NVCC) --version

clean:
	$(RM) -r build
