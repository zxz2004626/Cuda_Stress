#!/usr/bin/env bash
set -euo pipefail

CCBIN=/usr/bin/g++-11
ARCH=sm_86

printf "compiler opt checksum\n"
for tc in cuda-13.3 cuda-13.0 cuda-12.5; do
    nvcc=/export/ssd1/xzhanglb/nvcc-toolkit/$tc/bin/nvcc
    for opt in O0 O1 O2; do
        mkdir -p build
        exe=build/${tc}_${opt}
        "$nvcc" -w -std=c++17 -arch=$ARCH -ccbin "$CCBIN" -Xptxas=-$opt test.cu -o "$exe"
        out=$("$exe" | sed -n 's/^checksum=//p')
        printf "%s %s  %s\n" "$tc" "$opt" "$out"
    done
done
