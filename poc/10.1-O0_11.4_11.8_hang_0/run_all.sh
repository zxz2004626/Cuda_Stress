#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-"$THIS_DIR/build"}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-4}
SKIP_BUILD=${SKIP_BUILD:-0}
BINARY_NAME=${BINARY_NAME:-hang_0}

CUDA_10_1=${CUDA_10_1:-/export/d3/xzhanglb/nvcc-toolkit/cuda-10-1}
CUDA_11_4=${CUDA_11_4:-/export/d3/xzhanglb/nvcc-toolkit/cuda-11-4}
CUDA_11_8=${CUDA_11_8:-/export/d3/xzhanglb/nvcc-toolkit/cuda-11-8}
GXX8_LIB=${GXX8_LIB:-/export/d3/xzhanglb/nvcc-toolkit/gcc-8/usr/lib/x86_64-linux-gnu}

labels=(cuda-10.1 cuda-11.4 cuda-11.8)
roots=("$CUDA_10_1" "$CUDA_11_4" "$CUDA_11_8")
# Run the expected-success variants first.  A timed-out CUDA process can leave
# transient GPU-side state that makes later success cases look like false
# timeouts on this testcase.
run_order=(
  "0 1" "0 2" "0 3"
  "0 0"
  "1 0" "1 1" "1 2" "1 3"
  "2 0" "2 1" "2 2" "2 3"
)

if [[ "$SKIP_BUILD" != "1" ]]; then
  "$THIS_DIR/build_all.sh"
fi

printf "%-10s %-4s %-10s %s\n" "compiler" "opt" "status" "checksum_or_note"

for item in "${run_order[@]}"; do
  set -- $item
  i=$1
  opt=$2
  label=${labels[$i]}
  cuda_root=${roots[$i]}
  run_libs="$cuda_root/lib64"
  if [[ -d "$GXX8_LIB" ]]; then
    run_libs="$GXX8_LIB:$run_libs"
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    run_libs="$run_libs:$LD_LIBRARY_PATH"
  fi

  exe="$BUILD_DIR/$label/O$opt/$BINARY_NAME"
  out="$BUILD_DIR/$label/O$opt/run.out"
  err="$BUILD_DIR/$label/O$opt/run.err"
  if [[ ! -x "$exe" ]]; then
    printf "%-10s %-4s %-10s %s\n" "$label" "O$opt" "MISSING" "$exe"
    continue
  fi

  set +e
  LD_LIBRARY_PATH="$run_libs" timeout "${TIMEOUT_SECONDS}s" "$exe" >"$out" 2>"$err"
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    checksum=$(awk -F= '/^checksum=/{print $2}' "$out" | tail -n 1)
    if [[ -z "$checksum" ]]; then
      checksum="exit0"
    fi
    printf "%-10s %-4s %-10s %s\n" "$label" "O$opt" "OK" "$checksum"
  elif [[ $rc -eq 124 ]]; then
    printf "%-10s %-4s %-10s timeout_after_%ss\n" "$label" "O$opt" "TIMEOUT" "$TIMEOUT_SECONDS"
  else
    note=$(head -n 1 "$err" 2>/dev/null || true)
    printf "%-10s %-4s %-10s rc=%s %s\n" "$label" "O$opt" "ERROR" "$rc" "$note"
  fi
done
