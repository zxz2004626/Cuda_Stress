# miscompile-O1plus-fe535f18-cv15-indexcast

Follow-up variant discovered while reducing
`poc/miscompile-O1only-fe535f18/reduced.cu`.

Compared with the O1-only sibling, this variant removes only the outer
`(u)` cast from the `cv_15` input index:

```cpp
u cv_15 = input[GPU_SMOD(live_0 * 32ll + pc_r2_0, n)];
```

That changes the CUDA 13.3 `ptxas` differential from O1-only to O1/O2/O3.

## Expected Result

On CUDA 13.3 targeting `sm_86`:

| opt | status | checksum |
| --- | --- | --- |
| O0 | CORRECT | 413462180 |
| O1 | WRONG | 894037668 |
| O2 | WRONG | 894037668 |
| O3 | WRONG | 894037668 |

## Notes

This is saved as a separate research candidate rather than merged into the
O1-only reducer because it changes the optimization differential class. The
single source change affects ptxas' typed index/dataflow shape while preserving
the fixed-launch runtime address, since `n == 1`.

The single source change barely changes PTX: it only moves an independent
`mul.lo.s32` around the `cv_15` load address calculation. The runtime address
and the PTX semantics are unchanged. That tiny ordering perturbation is enough
to make ptxas O2/O3 pick the same faulty shift-count schedule already seen in
the O1-only O1 build.

The shared root cause is therefore still the final `_lc_15` call to `helper_1`.
Correct PTX computes `gridDim.y << threadIdx.y`; bad SASS computes
`gridDim.y << gridDim.y`, so the fixed launch shifts by 2 instead of 1 and
produces `_lc_15 == 0xa738d73f` instead of `0xabb4c6bf`.
