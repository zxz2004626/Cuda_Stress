#!/usr/bin/env bash
set -euo pipefail
SRC="${SRC:-$(dirname "$0")/test.cu}"
BUILD_DIR="${BUILD_DIR:-$(dirname "$0")/build}"
mkdir -p "$BUILD_DIR"
CCBIN="${CCBIN:-/usr/bin/g++-11}"
ARCH="${ARCH:-sm_86}"

declare -A NVCCS=(
  [cuda-13.3]="/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.3/bin/nvcc"
  [cuda-13.0]="/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.0/bin/nvcc"
  [cuda-12.5]="/export/ssd1/xzhanglb/nvcc-toolkit/cuda-12.5/bin/nvcc"
)

echo "compiler opt checksum"
for tc in cuda-13.3 cuda-13.0 cuda-12.5; do
  nvcc="${NVCCS[$tc]}"
  for opt in O0 O1 O2; do
    exe="$BUILD_DIR/${tc}_${opt}"
    "$nvcc" -w -std=c++17 -arch="$ARCH" -ccbin "$CCBIN" -Xptxas=-"$opt" "$SRC" -o "$exe"
    out=$(timeout 20s "$exe")
    checksum=${out#checksum=}
    printf '%-9s %-3s %s
' "$tc" "$opt" "$checksum"
  done
done
