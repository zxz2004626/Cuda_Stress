# miscompile-O1only-fe535f18

Self-contained reduced CUDA reproducer from
`generator_ir__fe535f18dab5a1bd`.

## Summary

CUDA 13.3 `ptxas` produces a different result only at `-Xptxas=-O1`.
`-O0`, `-O2`, and `-O3` agree on the reduced testcase.

## Usage

```bash
cd /export/ssd1/xzhanglb/Cuda_Stress/poc/miscompile-O1only-fe535f18
make
```

Useful overrides:

```bash
make NVCC=/path/to/cuda-13.3/bin/nvcc ARCH=sm_86
make clean
```

## Expected Result

On the reproducing CUDA 13.3 toolchain on `sm_86`, `make` produces:

| opt | status | checksum |
| --- | --- | --- |
| O0 | CORRECT | 6285142377 |
| O1 | WRONG | 5964855639 |
| O2 | CORRECT | 6285142377 |
| O3 | CORRECT | 6285142377 |

## UB Check

The reduced testcase uses `n=1`, one block of 1 thread, and only `tid == 0`
writes `output[0]`. The input buffer has 1 word and is initialized before the
kernel launch. The retained shared-memory handoff (`pc_comm_0[2]`) is explicitly
initialized before use and keeps one block-wide `__syncthreads()` barrier. The
retained device global `arr_0` is now `arr_0[16]` and is explicitly
zero-initialized with `cudaMemcpyToSymbol`.

Expected CUDA 13.3 `compute-sanitizer` results on the final sample:

| binary | memcheck | initcheck | synccheck | racecheck |
| --- | --- | --- | --- | --- |
| O0 | 0 errors | 0 errors | 0 errors | 0 hazards |
| O1 | 0 errors | 0 errors | 0 errors | 0 hazards |

The source is formatted for readability: 311 lines and 15.0 KiB.
