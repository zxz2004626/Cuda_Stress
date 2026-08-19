# deadloop_0

Self-contained CUDA runtime reproducer for the CUDA 10.1 `ptxas -O0` runtime
dead-loop issue.

This directory intentionally does not depend on the parent repository's Python
pipeline or `native/cuda_runner`.  The single `.cu` file contains both:

- the reduced no-inline CUDA kernel;
- a small CUDA Runtime host launcher that initializes input, launches the kernel,
  copies output back, and prints a deterministic checksum.

Default toolchain paths match this host:

- CUDA 10.1: `/export/d3/xzhanglb/nvcc-toolkit/cuda-10-1`
- CUDA 11.4: `/export/d3/xzhanglb/nvcc-toolkit/cuda-11-4`
- CUDA 11.8: `/export/d3/xzhanglb/nvcc-toolkit/cuda-11-8`
- GCC 8 for CUDA 10.1 compatibility:
  `/export/d3/xzhanglb/nvcc-toolkit/gcc-8/usr/bin/g++-8`

Build all 12 variants:

```bash
./build_all.sh
```

Run all 12 variants:

```bash
./run_all.sh
```

The matrix is:

- compiler versions: CUDA 10.1, CUDA 11.4, CUDA 11.8
- ptxas optimization levels: `-Xptxas -O0`, `-O1`, `-O2`, `-O3`

Useful environment overrides:

```bash
CUDA_10_1=/path/to/cuda-10.1 \
CUDA_11_4=/path/to/cuda-11.4 \
CUDA_11_8=/path/to/cuda-11.8 \
GXX8=/path/to/g++-8 \
TIMEOUT_SECONDS=5 \
DEVICE=0 \
./run_all.sh
```

Expected on the original failing setup: CUDA 10.1 with `-Xptxas -O0` times out;
optimized CUDA 10.1 and newer CUDA versions should finish and print checksums.

Observed on this host:

```text
compiler   opt  status     checksum_or_note
cuda-10.1  O0   TIMEOUT    timeout_after_5s
cuda-10.1  O1   OK         5510415734763028640
cuda-10.1  O2   OK         5510415734763028640
cuda-10.1  O3   OK         5510415734763028640
cuda-11.4  O0   OK         5510415734763028640
cuda-11.4  O1   OK         5510415734763028640
cuda-11.4  O2   OK         5510415734763028640
cuda-11.4  O3   OK         5510415734763028640
cuda-11.8  O0   OK         5510415734763028640
cuda-11.8  O1   OK         5510415734763028640
cuda-11.8  O2   OK         5510415734763028640
cuda-11.8  O3   OK         5510415734763028640
```

## Root cause from SASS

This is a CUDA 10.1 ptxas `-O0` backend control-flow/code-generation bug. The PTX embedded in the CUDA 10.1 `-O0` and `-O1` binaries is identical except for the recorded `ptxasOptions` string, so the dead loop is introduced after PTX, during ptxas SASS generation.

The relevant source construct is the side-effect-free 64-iteration loop in `helper_0`:

```cuda
for (int i_8 = 0; i_8 < 64; ++i_8) {
  (void)__popcll(__funnelshift_rc(p0, 13, 18));
}
```

In PTX, CUDA 10.1 represents this as a chunked loop whose induction variable is initialized to `-64`, then incremented by `16` until it reaches zero:

```ptx
mov.u32 %r174, -64;
BB0_6:
  ...
  add.s32 %r174, %r174, 16;
  setp.ne.s32 %p5, %r174, 0;
  @%p5 bra BB0_6;
```

With the correct initial value, the loop executes four chunks and exits:

```text
-64 -> -48 -> -32 -> -16 -> 0
```

The faulty CUDA 10.1 `-O0` SASS has an entry edge that jumps directly to the loop latch at `0x970`, bypassing the `-64` initialization:

```sass
/*0440*/ BSSY B0, 0x9c0 ;
/*0450*/ BRA 0x970 ;
...
/*0930*/ MOV R16, 0xffffffc0 ;
/*0950*/ @!P0 BRA 0x430 ;
/*0960*/ BRA 0x5e0 ;
/*0970*/ IADD3 R16, R16, 0x10, RZ ;
/*0980*/ ISETP.NE.AND P0, PT, R16, RZ, PT ;
/*09a0*/ @P0 BRA 0x970 ;
```

The initialization `MOV R16, 0xffffffc0` is present, but it is not on the first-entry path into the counter-only loop. Therefore `R16` can contain a stale value from prior arithmetic. If that stale value is not 16-aligned to zero modulo `2^32`, repeatedly adding `16` can never make it equal to zero. The latch then becomes a true infinite loop:

```text
R16 += 16;
while (R16 != 0) repeat;
```

Root cause summary: CUDA 10.1 ptxas `-O0` misplaces the induction-variable initialization while lowering a side-effect-free 64-iteration loop into a counter-only chunk loop. The generated SASS has a direct entry to the latch that bypasses the required `R16 = -64`, producing a non-terminating loop. Newer ptxas versions and CUDA 10.1 optimized modes do not generate this broken control-flow shape.
