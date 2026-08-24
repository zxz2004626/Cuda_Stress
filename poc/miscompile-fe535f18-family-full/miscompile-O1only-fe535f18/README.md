# miscompile-O1only-fe535f18

Self-contained reduced CUDA reproducer from
`generator_ir__fe535f18dab5a1bd`.

## Summary

CUDA 13.3 `ptxas` produces a different result only at `-Xptxas=-O1`.
`-O0`, `-O2`, and `-O3` agree on the reduced testcase.

There are two kept source forms:

- `reduced.cu`: 120-line readable reduced trigger, further reduced from the pasted input.
  This is the Makefile default.
- `test.cu`: 311-line readable, UB-hardened form used for sanitizer checks.

## Usage

```bash
cd poc/miscompile-O1only-fe535f18
make
```

Useful overrides:

```bash
make NVCC=/path/to/cuda-13.3/bin/nvcc ARCH=sm_86
make SRC=test.cu NVCC=/path/to/cuda-13.3/bin/nvcc ARCH=sm_86
make clean
```

## Expected Result

On the reproducing CUDA 13.3 toolchain on `sm_86`, default `make`
(`SRC=reduced.cu`) produces:

| opt | status | output |
| --- | --- | --- |
| O0 | CORRECT | 2053990987 |
| O1 | WRONG | 2534566475 |
| O2 | CORRECT | 2053990987 |
| O3 | CORRECT | 2053990987 |

For the readable sanitizer-oriented source (`SRC=test.cu`), the corresponding
checksums are:

| opt | status | checksum |
| --- | --- | --- |
| O0 | CORRECT | 6285142377 |
| O1 | WRONG | 5964855639 |
| O2 | CORRECT | 6285142377 |
| O3 | CORRECT | 6285142377 |

## UB Check

The readable `test.cu` testcase uses `n=1`, one block of 1 thread, and only
`tid == 0` writes `output[0]`. The input buffer has 1 word and is initialized
before the kernel launch. The retained shared-memory handoff (`pc_comm_0[2]`) is
explicitly initialized before use and keeps one block-wide `__syncthreads()`
barrier. The retained device global `arr_0` is `arr_0[16]` and is explicitly
zero-initialized with `cudaMemcpyToSymbol`.

Expected CUDA 13.3 `compute-sanitizer` results on the final sample:

| binary | memcheck | initcheck | synccheck | racecheck |
| --- | --- | --- | --- | --- |
| O0 | 0 errors | 0 errors | 0 errors | 0 hazards |
| O1 | 0 errors | 0 errors | 0 errors | 0 hazards |

The sanitizer-checked source is formatted for readability: 311 lines and
15.0 KiB. The default reduced trigger is kept readable at 120 lines and
5.0 KiB.

## Root Cause Boundary

The failure is a CUDA 13.3 `ptxas` backend optimization bug, not CUDA source
undefined behavior in the sanitizer-checked form. Generating PTX with
`nvcc -O0`, `-O1`, `-O2`, and `-O3` gives the same PTX; changing only the
`ptxas` optimization level to `-Xptxas=-O1` changes the result.

The first wrong value is produced by the final `_lc_15` update:

```cpp
_lc_15 ^= helper_1(_lc_0, 0, 0, _lc_3);
```

The PTX for `helper_1` is correct: it computes the nested shift count as
`gridDim.y << threadIdx.y` and then shifts `p3` by that count. With the fixed
launch `<<<1, 1>>>`, the intended shift count is `1 << 0 == 1`.

The bad O1 SASS overwrites the `threadIdx.y` register with `gridDim.y` before
the shift-count `SHF`:

```sass
S2R         R9, SR_TID.Y
ISETP.GE   P1, PT, R9, 0x20, PT
IMAD.MOV   R9, RZ, RZ, c[0x0][0x10]
SHF.L.U32  R9, R9, R9, RZ        // computes gridDim.y << gridDim.y
SHF.L.U32  R9, R5, R9, RZ        // shifts p3 by 2, not 1
```

That changes the first `_lc_15` value from `0xabb4c6bf` to `0xa738d73f`,
which is exactly the first observed raw-output divergence. O2/O3 use a different
register/spill layout and preserve the correct shift count, so they agree with
O0 in this variant.

The remaining trigger is context-sensitive rather than one isolated arithmetic
operation. The legacy `__mul24` contribution, global-derived remainder path,
helper calls, fallthrough switch, launch-bounds metadata, and enough unrolled
integer-mixer live state are shape factors that steer ptxas into or away from
the faulty register/scheduling layout.
Even with the fixed single-thread launch and fixed `n=1, var_0=1, input[0]=12345`,
many individually constant-valued loads still have to remain as loads; replacing
individual `R(tid+k)` uses in the `rs_v`, `b*`, or `j*` paths with `12345`
removes the O1-only divergence.
The default `reduced.cu` is not constrained to preserve the original output:
the retained condition is that `-Xptxas=-O0/-O2/-O3` agree and `-Xptxas=-O1`
differs. A later pass simplified the switch selector to `threadIdx.x & 3`;
the empty `case 3` label is still required for this control-flow shape.
Further remote reduction showed that removing `__launch_bounds__(512, 2)`,
lowering it to `64/128/256, 2`, or using `512, 1` also makes the divergence
disappear; `512, 3` and `1024, 2` still reproduce.

## Practical Avoidance

For production kernels, avoid using `__mul24` inside complex mixed-width integer
mixers unless 24-bit multiply semantics are required. Prefer ordinary 32-bit
multiply on masked values, or split the expression so the 24-bit multiply,
64-bit `%`, helper calls, and long live-state mixer are not all in one ptxas O1
optimization region. As a compiler workaround for affected kernels, compile
that translation unit or kernel with `-Xptxas=-O2` (or `-Xptxas=-O0` if the
performance tradeoff is acceptable).

Also avoid the nested dynamic shift-count pattern
`GPU_SSHL(p3, GPU_SSHL(gridDim.y, threadIdx.y))` in highly constrained
launch-bounds kernels on CUDA 13.3. Materializing the inner count in a simpler
form or splitting the helper can avoid this backend path.
