# miscompile-O23plus-fe535f18-aggressive

Aggressive follow-up reducer derived from
`poc/miscompile-O1only-fe535f18/reduced.cu`.

This variant intentionally folds the single-thread shared-memory handoff:

```cpp
u pc_val_0 = R(tid), pc_r2_0 = 0;
u pc_r3_0 = 0;
```

The fold changes the optimizer shape enough that CUDA 13.3 `ptxas` now
miscompiles at `-Xptxas=-O2` and `-Xptxas=-O3`, while `-O0` and `-O1` agree.

## Usage

```bash
cd poc/miscompile-O23plus-fe535f18-aggressive
make NVCC=/path/to/cuda-13.3/bin/nvcc ARCH=sm_86
```

## Expected Result

On CUDA 13.3 targeting `sm_86`:

| opt | status | checksum |
| --- | --- | --- |
| O0 | CORRECT | 4096945828 |
| O1 | CORRECT | 4096945828 |
| O2 | WRONG | 1225920164 |
| O3 | WRONG | 1225920164 |

## Root Cause Boundary

This is still a CUDA 13.3 `ptxas` backend miscompile, not a source-level
requirement to use shared memory. The folded scalar version is valid for the
fixed launch `<<<1, 1>>>`, because the removed shared-memory sequence stored
zero then read the same single element.

This variant hits the same bad instruction selection/scheduling pattern as the
O1-only sibling, but at `-Xptxas=-O2/-O3`. The first wrong value is again the
final `_lc_15` update:

```cpp
_lc_15 ^= helper_1(_lc_0, 0, 0, _lc_3);
```

The PTX computes `gridDim.y << threadIdx.y` before using that as the shift
count for `p3`. The bad O2/O3 SASS instead clobbers the `threadIdx.y` register
with `gridDim.y` and computes `gridDim.y << gridDim.y`:

```sass
S2R         R20, SR_TID.Y
ISETP.GE   P4, PT, R20, 0x20, PT
IMAD.MOV   R20, RZ, RZ, c[0x0][0x10]
SHF.L.U32  R20, R20, R20, RZ
SHF.L.U32  R10, R2, R20, RZ
```

For the fixed launch, this shifts by 2 instead of 1. The traced `_lc_15`
therefore changes from `0xabb4c6bf` to `0xa738d73f`, and the checksum diverges
after the `_lc_*` tail mix.

The high-optimization trigger is narrower than the O1-only sibling:

- Removing `__launch_bounds__(512, 2)` makes O0/O1/O2/O3 agree.
- Replacing the fallthrough `switch (tid & 3)` with the folded constant
  `1567090485` makes O0/O1/O2/O3 agree.
- Replacing the fallthrough `switch` with the equivalent straight-line
  fallthrough computation makes O0/O1/O2/O3 agree.
- `__launch_bounds__(512, 2)` and `__launch_bounds__(1024, 2)` reproduce at
  O2/O3. After the latest reduction, the default sample compiles O2/O3 to
  64 registers and 112 bytes stack.
- `__launch_bounds__(512)`, `__launch_bounds__(512, 1)`,
  `__launch_bounds__(256, 2)`, and `__launch_bounds__(512, 3)` do not reproduce.
- Applying `--maxrregcount=32..128` without launch-bounds metadata does not
  reproduce, even when ptxas is forced to 64 registers.

So the practical trigger boundary is the interaction between launch-bounds
occupancy metadata, ptxas O2/O3 scheduling/spill layout, and the retained
fallthrough switch/live-range shape. The shared-memory handoff in the O1-only
sibling incidentally steered O2/O3 away from this faulty backend layout; folding
it and then deleting the now-zero pc mixes exposes the same wrong shift-count
path to O2/O3.

## Practical Avoidance

For affected CUDA 13.3 kernels, avoid combining strong launch bounds such as
`__launch_bounds__(512, 2)` with complex long-live integer mixer code and
fallthrough switches. If the code is fixed to a single-thread debug/reproducer
launch, do not assume that manually folding memory handoffs will preserve which
ptxas optimization levels are affected.

The most direct source-level pattern to avoid is the nested shift count
`GPU_SSHL(p3, GPU_SSHL(gridDim.y, threadIdx.y))` inside the unrolled helper.
