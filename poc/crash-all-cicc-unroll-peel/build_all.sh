#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-"$THIS_DIR/test.cu"}
BUILD_DIR=${BUILD_DIR:-"$THIS_DIR/build"}
ARCH=${ARCH:-sm_75}

# Default toolchains: shuaicpu4 (shuaicpu4.cse.ust.hk). Override via env.
CUDA_11_5_NVCC=${CUDA_11_5_NVCC:-/usr/bin/nvcc}
CUDA_12_5=${CUDA_12_5:-/export/ssd1/xzhanglb/nvcc-toolkit/cuda-12.5}
CUDA_13_0=${CUDA_13_0:-/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.0}
CUDA_13_1=${CUDA_13_1:-/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.1}
CUDA_13_2=${CUDA_13_2:-/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.2}
CUDA_13_3=${CUDA_13_3:-/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.3}

labels=(cuda-11.5 cuda-12.5 cuda-13.0 cuda-13.1 cuda-13.2 cuda-13.3)
nvccs=("$CUDA_11_5_NVCC" "$CUDA_12_5/bin/nvcc" "$CUDA_13_0/bin/nvcc" \
       "$CUDA_13_1/bin/nvcc" "$CUDA_13_2/bin/nvcc" "$CUDA_13_3/bin/nvcc")
opts=(0 1 2 3)

mkdir -p "$BUILD_DIR"
manifest="$BUILD_DIR/manifest.tsv"
printf "compiler\topt\trc\tstatus\tptx\n" >"$manifest"

for i in "${!labels[@]}"; do
  label=${labels[$i]}
  nvcc=${nvccs[$i]}
  if [[ ! -x "$nvcc" ]]; then
    echo "missing nvcc for $label: $nvcc" >&2
    printf "%s\t-\t-\tMISSING_TOOLCHAIN\t-\n" "$label" >>"$manifest"
    continue
  fi
  for opt in "${opts[@]}"; do
    out_dir="$BUILD_DIR/$label/O$opt"
    mkdir -p "$out_dir"
    ptx="$out_dir/test.ptx"
    log="$out_dir/compile.log"
    set +e
    "$nvcc" -std=c++17 -arch="$ARCH" -O"$opt" -ptx -o "$ptx" "$SRC" >"$log" 2>&1
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      status=OK
    elif [[ $rc -ge 128 || $rc -lt 0 ]]; then
      status=CRASH
    else
      status="ERR$rc"
    fi
    echo "COMPILE $label -O$opt -> $status (rc=$rc)"
    printf "%s\tO%s\t%s\t%s\t%s\n" "$label" "$opt" "$rc" "$status" "$ptx" >>"$manifest"
  done
done

echo "compiled ${#labels[@]} toolchains x ${#opts[@]} opt levels under $BUILD_DIR"
