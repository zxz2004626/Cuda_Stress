# deadloop_1

Self-contained CUDA-C reproducer reduced from:

`poc/cuda-10.1_O0_deadloop/generator_ir__5fc52dba44776f66/test.cu`

It does not use inline PTX and does not depend on the `cuda_stress` Python package or other repository code. The scripts only assume local CUDA toolkit installs.

## Files

- `runtime_minimal_no_inline.cu`: reduced CUDA source.
- `build_all.sh`: builds CUDA 10.1 / 11.4 / 11.8 across ptxas `-O0..-O3`, and dumps SASS beside each binary.
- `run_all.sh`: builds by default, then runs the 12 binaries with a timeout and prints a matrix.

Default toolchain paths:

- `CUDA_10_1=/export/d3/xzhanglb/nvcc-toolkit/cuda-10-1`
- `CUDA_11_4=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-4`
- `CUDA_11_8=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-8`
- `GXX8=/export/d3/xzhanglb/nvcc-toolkit/gcc-8/usr/bin/g++-8`

Override them through environment variables if needed. The default target architecture is `sm_75`; override with `ARCH=...`.

## Usage

```bash
cd /export/d3/xzhanglb/Cuda_Stress/poc/10.1_deadloop_1
./build_all.sh
SKIP_BUILD=1 ./run_all.sh
```

or simply:

```bash
./run_all.sh
```

## Observed result on this host

Timeout is set to 5 seconds.

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | TIMEOUT | OK checksum=0 | OK checksum=0 | OK checksum=0 |
| cuda-11.4 | OK checksum=0 | OK checksum=0 | OK checksum=0 | OK checksum=0 |
| cuda-11.8 | OK checksum=0 | OK checksum=0 | OK checksum=0 | OK checksum=0 |

## Reduction notes

The minimized trigger is CUDA source only. The necessary shape retained here is:

- one 1-thread kernel launch;
- two calls to `helper_0` whose results are both consumed;
- a guarded division feeding the `i_3` loop bound;
- a small `do/while` loop resetting `p0`;
- the 64-iteration `i_5` loop containing a `threadIdx.x`-dependent shift expression.

Single-call variants and several constant-folded forms (`__brev` branch folded, `i_5` shift folded, or replacing the `do/while` with plain assignments) stop reproducing the CUDA 10.1 `-Xptxas -O0` runtime deadloop.

## Root cause from SASS

This is the same bug family as `10.1_deadloop_0`: CUDA 10.1 ptxas `-O0` generates a counter-only loop whose induction-variable initialization is not on all entry paths. The PTX embedded in the CUDA 10.1 `-O0` and `-O1` binaries is identical except for `ptxasOptions`, so the issue is introduced by ptxas SASS generation rather than the CUDA front-end.

The source trigger is the 64-iteration loop in `helper_0`:

```cuda
for (int i_5 = 0; i_5 < 64; i_5 += 1) {
  p1 = 4294967231u;
  p0 = (((unsigned long long)p1 << (shift & 31u)) ^
        (3771765082u >> (shift & 31u)));
}
```

PTX lowers this to repeated 16-iteration chunks, using a counter initialized to `-64` and incremented by `16` until zero. CUDA 10.1 `-O0` expands both `helper_0(0u, 0u)` calls in SASS. The first call exits, but the second call has a direct path into the chunk loop that bypasses the `-64` initialization.

The faulty path is visible in the CUDA 10.1 `-O0` SASS:

```sass
/*1c90*/ @P1 BRA 0x1f70 ;
...
/*1f80*/ BSYNC B0 ;
/*1f90*/ BSSY B0, 0x35d0 ;
/*1fa0*/ BRA 0x2400 ;
...
/*23c0*/ MOV R2, 0xffffffc0 ;
...
/*3570*/ IADD3 R2, R2, 0x10, RZ ;
/*3580*/ ISETP.NE.AND P0, PT, R2, RZ, PT ;
/*35b0*/ @P0 BRA 0x2400 ;
```

The intended initialization is `MOV R2, 0xffffffc0`, i.e. `R2 = -64`. It is present in the SASS, but the taken branch path goes through `0x1f70 -> 0x1f90 -> 0x2400`, so it can enter the chunk loop without executing the initialization at `0x23c0`.

For this minimized one-thread testcase, the stale `R2` value on that path comes from the earlier `p1 = p0 * 1338290939u` computation and is not congruent to zero modulo `16` after repeated `+16` updates. Therefore the loop latch:

```text
R2 += 16;
while (R2 != 0) repeat;
```

never reaches zero and the kernel does not return. With the intended initialized value, the sequence would be finite:

```text
-64 -> -48 -> -32 -> -16 -> 0
```

Root cause summary: CUDA 10.1 ptxas `-O0` incorrectly places the induction-variable initialization for the chunked 64-iteration loop. A valid control-flow path in the second inlined `helper_0` call bypasses `R2 = -64`, so the latch operates on a stale, non-terminating value. CUDA 10.1 `-O1..-O3` and CUDA 11.4/11.8 do not generate this broken SASS.
