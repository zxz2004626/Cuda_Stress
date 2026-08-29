# miscompile-fe535f18-family

This directory groups four closely related CUDA 13.3 `ptxas` wrong-result
reducers derived from `generator_ir__fe535f18dab5a1bd`.

All four variants share the same semantic root cause: PTX computes the nested
shift count in `helper_1` as `gridDim.y << threadIdx.y`, but a bad ptxas
register/scheduling layout can generate SASS that effectively computes
`gridDim.y << gridDim.y`. With the fixed launch `<<<1, 1>>>`, the correct count
is `1`, while the bad SASS uses `2`.

The common source pattern is:

```cpp
p0 -= GPU_SSHL(p3 - i_0, GPU_SSHL(gridDim.y, threadIdx.y));
```

and the first observed wrong value is reached through the tail call:

```cpp
_lc_15 ^= helper_1(_lc_0, 0, 0, _lc_3);
```

## Variants

| file | differential | correct output | wrong output | purpose |
| --- | --- | --- | --- | --- |
| `o1only.cu` | O1 wrong; O0/O2/O3 correct | `2053990987` | `2534566475` | Baseline readable reducer. |
| `o1plus-cv15-indexcast.cu` | O1/O2/O3 wrong; O0 correct | `2053990987` | `2534566475` | One-line index-cast perturbation. |
| `o23plus-aggressive.cu` | O2/O3 wrong; O0/O1 correct | `1442507339` | `2866448971` | Aggressively folded shared-memory/pc path. |
| `o23plus-i1trip.cu` | O2/O3 wrong; O0/O1 correct | `4123129483` | `3429195403` | Minimal loop-bound perturbation from O1-only. |

## Usage

Run the baseline variant:

```bash
make NVCC=/path/to/cuda-13.3/bin/nvcc ARCH=sm_86
```

Run a specific variant:

```bash
make VARIANT=o23plus-aggressive NVCC=/path/to/cuda-13.3/bin/nvcc ARCH=sm_86
```

Run all four:

```bash
make run-all NVCC=/path/to/cuda-13.3/bin/nvcc ARCH=sm_86
```

List variant names:

```bash
make variants
```

## Relationship

`o1only.cu` is the baseline shape. Its O1 SASS contains the clearest bad
sequence:

```sass
S2R         R9, SR_TID.Y
ISETP.GE   P1, PT, R9, 0x20, PT
IMAD.MOV   R9, RZ, RZ, c[0x0][0x10]
SHF.L.U32  R9, R9, R9, RZ
SHF.L.U32  R9, R5, R9, RZ
```

The third instruction overwrites the register that previously held
`threadIdx.y` with `gridDim.y`, so the fourth instruction computes
`gridDim.y << gridDim.y`.

`o1plus-cv15-indexcast.cu` differs from `o1only.cu` by removing only the outer
`(u)` cast from the `cv_15` index. Runtime addressing is unchanged for `n == 1`,
and PTX changes only by a tiny independent instruction ordering perturbation.
That perturbation is nevertheless enough for ptxas O2/O3 to choose the same bad
shift-count layout already seen at O1.

`o23plus-aggressive.cu` folds the single-thread shared-memory handoff and drops
now-zero pc mixes. This is not equivalent as a general CUDA program, but it is
valid for the fixed reproducer launch. The fold changes live ranges enough that
O2/O3, rather than O1, select the bad backend layout.

`o23plus-i1trip.cu` keeps the shared-memory handoff but changes the inner mixer
loop bound from `i_1 < 4` to `i_1 < 3`. It shows the same family effect with a
much smaller source perturbation: a small live-range/control-shape change moves
the failure from O1-only to O2/O3.

## Notes

The fallthrough switch, launch bounds, helper control flow, unrolled input
mixer, and tail `_lc_*` state are mostly trigger-shape anchors. Many are not the
first wrong arithmetic operation, but deleting or constant-folding them changes
ptxas register allocation enough to hide the bug.
