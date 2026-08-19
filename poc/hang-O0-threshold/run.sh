#!/usr/bin/env bash
# UNREDUCED STAND-IN — see recovery note in README.md.
#
# The reduced `test.cu` (which reproduces the clean O0-only timeout split under a
# 10s threshold) was lost in a concurrent `git clean`. This script instead builds and
# times the *original, unreduced* generator source, which on this host times out at
# O0 AND O1 (it does not by itself show the O0-only split). It is kept as the starting
# point for re-reduction.
set -euo pipefail

THIS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC="${SRC:-$THIS_DIR/../../runs/run-20260819-000123-2433/differentials/generator_ir__418864b9bd901c35/test.cu}"
BUILD_DIR="${BUILD_DIR:-$THIS_DIR/build}"
RUN_TIMEOUT="${RUN_TIMEOUT:-13}"
ARCH="${ARCH:-sm_86}"
CCBIN="${CCBIN:-/usr/bin/g++-11}"

declare -A NVCCS=(
  [cuda-13.3]="/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.3/bin/nvcc"
  [cuda-13.0]="/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.0/bin/nvcc"
  [cuda-12.5]="/export/ssd1/xzhanglb/nvcc-toolkit/cuda-12.5/bin/nvcc"
)

mkdir -p "$BUILD_DIR"

echo "NOTE: unreduced stand-in source (reduced test.cu was lost; see README recovery note)."
echo "compiler opt status elapsed"
for tc in cuda-13.3 cuda-13.0 cuda-12.5; do
  nvcc="${NVCCS[$tc]}"
  for opt in O0 O1 O2; do
    exe="$BUILD_DIR/${tc}_${opt}"
    "$nvcc" -w -std=c++11 -arch="$ARCH" -ccbin "$CCBIN" -Xptxas=-"$opt" "$SRC" -o "$exe"
    start=$(date +%s.%N)
    if timeout "${RUN_TIMEOUT}s" "$exe" >"$exe.out" 2>&1; then
      rc=0
    else
      rc=$?
    fi
    end=$(date +%s.%N)
    elapsed=$(echo "$end $start" | awk '{printf "%.1f", $1-$2}')
    if [[ $rc -eq 124 ]]; then
      status="timeout"
    elif [[ $rc -eq 0 ]]; then
      status="complete"
    else
      status="error"
    fi
    printf '%-9s %-3s %-8s %ss\n' "$tc" "$opt" "$status" "$elapsed"
  done
done
