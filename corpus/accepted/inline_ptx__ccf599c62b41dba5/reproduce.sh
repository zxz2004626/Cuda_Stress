#!/usr/bin/env bash
# Byte-for-byte reproduction of the ptxas O1-vs-O0 divergence case.
#
#   inline_ptx__ccf599c62b41dba5
#   strategy inline_ptx, config_seed=20260810, test_seed=9999
#
# Compiles the SAME source at ptxas -O0..-O3, runs each on the GPU and prints whether
# the output matches the CPU oracle.  Expected on cuda-11.8/12.1/12.6:
#     O0: MATCH,  O1/O2/O3: DIVERGE
# (cuda-11.3 stays correct at all levels.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CID=inline_ptx__ccf599c62b41dba5
NVCC=${NVCC:-/usr/local/cuda-12.6/bin/nvcc}
ARCH=sm_86
OUT=/tmp/cuda_stress_repro_$CID
RUNNER=$ROOT/native/cuda_runner

echo "regenerating $CID ..."
python3 -m cuda_stress reproduce --id "$CID" --out "$OUT" >/dev/null
cd "$OUT/$CID"

for O in 0 1 2 3; do
    $NVCC -arch=$ARCH -cubin -Xptxas=-O$O -o k$O.cubin test.cu
    $RUNNER --cubin k$O.cubin --kernel test_kernel \
        --input input.bin --output out$O.bin --result exec$O.json \
        --input-words 256 --output-words 128 --grid 1 --block 64 \
        --seed 3970868306842774015 --guards 8 --test-id "$CID" --timeout-ms 60000
    # strip the 40-byte header and 8-byte tail canary before comparing
    if cmp -s <(dd if=out$O.bin bs=1 skip=40 count=1024 status=none) <(dd if=expected.bin bs=1 status=none); then
        echo "  -O$O : MATCH   (output == CPU oracle)"
    else
        echo "  -O$O : DIVERGE (output != CPU oracle)"
    fi
done
echo
echo "compare O0 vs O1 meaningful words:"
if cmp -s <(dd if=out0.bin bs=1 skip=40 count=1024 status=none) <(dd if=out1.bin bs=1 skip=40 status=none); then
    echo "  O0 == O1 (no divergence on this toolchain)"
else
    echo "  O0 != O1 (ptxas optimization changed the result)"
fi
