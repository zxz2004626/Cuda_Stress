#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-"$THIS_DIR/runtime_minimal_no_inline.cu"}
BUILD_DIR=${BUILD_DIR:-"$THIS_DIR/build"}
ARCH=${ARCH:-sm_75}

CUDA_10_1=${CUDA_10_1:-/export/d3/xzhanglb/nvcc-toolkit/cuda-10-1}
CUDA_11_4=${CUDA_11_4:-/export/d3/xzhanglb/nvcc-toolkit/cuda-11-4}
CUDA_11_8=${CUDA_11_8:-/export/d3/xzhanglb/nvcc-toolkit/cuda-11-8}
GXX8=${GXX8:-/export/d3/xzhanglb/nvcc-toolkit/gcc-8/usr/bin/g++-8}
GXX8_LIB=${GXX8_LIB:-/export/d3/xzhanglb/nvcc-toolkit/gcc-8/usr/lib/x86_64-linux-gnu}

labels=(cuda-10.1 cuda-11.4 cuda-11.8)
roots=("$CUDA_10_1" "$CUDA_11_4" "$CUDA_11_8")
opts=(0 1 2 3)

mkdir -p "$BUILD_DIR"
manifest="$BUILD_DIR/manifest.tsv"
printf "compiler\topt\texe\tsass\n" >"$manifest"

for i in "${!labels[@]}"; do
  label=${labels[$i]}
  cuda_root=${roots[$i]}
  nvcc="$cuda_root/bin/nvcc"
  cuobjdump="$cuda_root/bin/cuobjdump"

  if [[ ! -x "$nvcc" ]]; then
    echo "missing nvcc for $label: $nvcc" >&2
    exit 1
  fi

  ccbin_args=()
  if [[ -x "$GXX8" ]]; then
    ccbin_args=(-ccbin "$GXX8")
  fi

  tool_libs="$cuda_root/lib64"
  if [[ -d "$GXX8_LIB" ]]; then
    tool_libs="$GXX8_LIB:$tool_libs"
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    tool_libs="$tool_libs:$LD_LIBRARY_PATH"
  fi

  for opt in "${opts[@]}"; do
    out_dir="$BUILD_DIR/$label/O$opt"
    exe="$out_dir/deadloop_0"
    sass="$out_dir/deadloop_0.sass"
    mkdir -p "$out_dir"
    echo "BUILD $label -Xptxas -O$opt"
    LD_LIBRARY_PATH="$tool_libs" "$nvcc" -std=c++11 -arch="$ARCH" \
      -Xptxas "-O$opt" "${ccbin_args[@]}" "$SRC" -o "$exe"
    if [[ -x "$cuobjdump" ]]; then
      "$cuobjdump" --dump-sass "$exe" >"$sass"
    else
      : >"$sass"
    fi
    printf "%s\tO%s\t%s\t%s\n" "$label" "$opt" "$exe" "$sass" >>"$manifest"
  done
done

echo "built 12 variants under $BUILD_DIR"
