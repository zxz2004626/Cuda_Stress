#!/usr/bin/env bash
# Byte-for-byte reproduction of the over-MSB bfe/bfi ptxas O1+ divergence.
#   inline_ptx__ebf45243624dc5d6   (inline_ptx, config_seed=20260810, test_seed=777)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CID=inline_ptx__ebf45243624dc5d6
NVCC=${NVCC:-/usr/local/cuda-12.6/bin/nvcc}
ARCH=sm_86
OUT=/tmp/cuda_stress_repro_$CID
RUNNER=$ROOT/native/cuda_runner
python3 -m cuda_stress reproduce --id "$CID" --out "$OUT" >/dev/null
cd "$OUT/$CID"
for O in 0 1 2 3; do
    $NVCC -arch=$ARCH -cubin -Xptxas=-O$O -o k$O.cubin test.cu
    $RUNNER --cubin k$O.cubin --kernel test_kernel         --input input.bin --output out$O.bin --result exec$O.json         --input-words 256 --output-words 128 --grid 1 --block 64         --seed 8337606540844322850 --guards 8 --test-id "$CID" --timeout-ms 60000
    if cmp -s <(dd if=out$O.bin bs=1 skip=40 count=1024 status=none) <(dd if=expected.bin bs=1 status=none); then
        echo "  -O$O : MATCH   (output == CPU oracle)"
    else
        echo "  -O$O : DIVERGE (output != CPU oracle)"
    fi
done
