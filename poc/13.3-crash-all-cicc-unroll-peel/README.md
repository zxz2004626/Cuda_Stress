# crash-all-cicc-unroll-peel

Self-contained CUDA-C compile-crash reproducer reduced from:

`runs/case_dbab15f68f04b25c_matrix/test.cu` (545 lines -> 20 lines)

The bug is a **compile-time** crash: `nvcc` (the `cicc` device-compiler phase)
segfaults while compiling `test.cu` on CUDA 12.5 and later, while CUDA 11.5
compiles it cleanly. No GPU and no program execution are needed to reproduce —
compiling to PTX is sufficient. The case does not depend on the `cuda_stress`
Python package or any other repository code.

## Files

- `test.cu`: reduced CUDA source (the full trigger; do not simplify, see below).
- `Makefile`: self-contained build entry point using one `nvcc`. It asserts that
  the selected compiler is between CUDA 12.5 and 13.3 before compiling.

The default target architecture is `sm_75`; override it with `ARCH=...` (the
crash is arch-independent: `sm_50`..`sm_90` and `compute_75` all crash).

## Usage

```bash
cd poc/crash-all-cicc-unroll-peel
make                       # equivalent to: make OPT=0
# Try another optimization level (all O0..O3 are affected):
make clean && make OPT=3
```

`NVCC` may be set to a specific toolkit, for example
`make NVCC=/export/ssd1/xzhanglb/nvcc-toolkit/cuda-12.5/bin/nvcc`. The
`version-assert` target rejects toolchains outside CUDA 12.5–13.3.

One-shot manual check:

```bash
make NVCC=/export/ssd1/xzhanglb/nvcc-toolkit/cuda-12.5/bin/nvcc
                                                            # Segmentation fault in cicc (rc=139)
```

## Observed result

CUDA 12.5, 13.0, 13.1, 13.2, and 13.3 all reproduce a SIGSEGV inside `cicc`
(`nvcc` rc=139) at every `-O0..-O3` level tested. CUDA 11.5 compiles the same
source cleanly, but is intentionally rejected by the Makefile's version
assertion. `-G` (device debug) is the only nvcc mode that avoids the crash;
`-lineinfo` still crashes. 12.0..12.4 were not tested (toolkits unavailable).

## Trigger anatomy (every element is required)

- a `__device__` helper called **from a loop in the kernel** (>= 2 calls); a
  single call gets inlined and does not trigger;
- outer loop: `#pragma unroll N` with **N in {6, 7, 8}** and a **runtime,
  block-uniform bound** (constant bounds, N <= 5 or N >= 9 do not trigger);
- a runtime block-uniform `if` inside the outer loop (`if (true)` or divergent
  `threadIdx.x` conditions do not trigger);
- inner loop: `#pragma unroll 2` (factor exactly 2), constant **even** bound
  >= 4, and a runtime-uniform `break`;
- a `__threadfence()` after the inner loop (inside or outside the `if`).

## Root cause (from gdb + disassembly of cicc 12.5)

The crash is a NULL-pointer dereference in `cicc`'s loop unroll/peel
bookkeeping:

- crash site: `mov 0x18(%rax),%r14` with `rax=0` (0x12be629 in cuda-12.5
  cicc; the original 545-line case crashes at the same address);
- the immediately preceding call is a hash-map lookup (bucket-chain walk) that
  returns NULL — no loop-info entry for the outer `#pragma unroll` loop after
  the nested partially-unrolled loop + `membar` transformed the CFG;
- the caller dereferences the NULL node's child-loop vector (offsets 0x18/0x20)
  without a null check, testing membership in an open-addressing set
  (-1/-2 sentinels) and collecting into a `SmallVector<...,16>`;
- strings in the binary locate this machinery: `.peel`, `.peel.begin`,
  `.peel.newph`, `.peel.next`, `Failed : cannot do loop peeling`,
  `max-pragma-upperbound-unroll`.

CUDA 11.5 handles the same construct by fully unrolling with a trip-count
dispatch and completes normally; its cicc contains the same bug family (a more
complex input crashes it with the identical instruction pattern), so 12.5's
rewritten unroll/peel machinery moved the trigger window onto this common
shape rather than introducing a brand-new bug.

## Workarounds

Any one of these breaks the trigger: outer unroll factor <= 5 or >= 9,
constant outer bound, odd inner bound, removing the `__threadfence()`, or
compiling with `-G` (debug only).
