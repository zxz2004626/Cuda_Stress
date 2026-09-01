# CUDA 12.5 cicc front-end miscompile: dropped nonzero addend (bogus zero propagation)

Front-end (`cicc`/NVVM optimizer) differential originally found at:

`/export/ssd1/xzhanglb/Cuda_Stress/runs/run-20260829-011020-1bbc/differentials/generator_ir__0948c44ea19f0120/test.cu`

This is **not** a ptxas bug: the 12.5 PTX alone, JITed by the host driver
(`nvcc -arch=compute_86`), already produces the wrong output, and `nvcc -G`
on 12.5 is correct.

## Reproduce

```sh
cd /export/ssd1/xzhanglb/Cuda_Stress/poc/miscompile-all-cicc-dropaddend0
./run.sh
```

Observed output (RTX 3090, sm_86):

```text
compiler opt checksum
cuda-13.3 O0  15181709146300232064
cuda-13.3 O1  15181709146300232064
cuda-13.3 O2  15181709146300232064
cuda-13.2 O0  15181709146300232064
cuda-13.2 O1  15181709146300232064
cuda-13.2 O2  15181709146300232064
cuda-13.1 O0  15181709146300232064
cuda-13.1 O1  15181709146300232064
cuda-13.1 O2  15181709146300232064
cuda-13.0 O0  15181709146300232064
cuda-13.0 O1  15181709146300232064
cuda-13.0 O2  15181709146300232064
cuda-12.6 O0  15181709146300232064
cuda-12.6 O1  15181709146300232064
cuda-12.6 O2  15181709146300232064
cuda-12.5 O0  13004433893898885072
cuda-12.5 O1  13004433893898885072
cuda-12.5 O2  13004433893898885072
```

12.5 is the **last bad** version; 12.6 is the first fixed version. The bug is
independent of the ptxas optimization level (it is already present in the PTX).

## Test program

`test.cu` is a greedy source-level reduction of the original 435-line
generator case (261 lines). It is a self-contained checksum harness:
deterministic inputs, one kernel launch, host-side checksum.

> **Warning — reducer artifact elsewhere.** The IR-level reducer output at
> `reductions/generator_ir__0948c44ea19f0120_ir_reduced/` is NOT faithful:
> it shrank `helper_1` to an empty non-void function (falling off the end =
> UB), and its divergence comes from the front ends folding that UB
> differently — giving `helper_1` any defined body makes all toolchains
> agree. Do not use it to study this bug.

## PTX evidence

The only structural difference between the normalized 12.5 and 13.0 PTX
(see `asm/`), in the region that computes a bound compared against
`%nctaid.y`:

Correct CUDA 13.0 (`asm/cuda-13.0.ptx`):

```ptx
selp.b32  %r52, %r44, 2039404531, %p3;   ; v_a (runtime, nonzero on path)
selp.b32  %r53, 0, %r51, %p3;            ; v_b
shl.b32   %r54, %r53, 2;
xor.b32   %r56, %r54, 9;
not.b32   %r57, %r52;                    ; ~v_a
add.s32   %r58, %r53, %r57;              ; v_b + ~v_a
sub.s32   %r4, %r58, %r56;               ; v_b + ~v_a - ((v_b<<2) ^ 9)
setp.lt.u32 %p4, %nctaid.y, %r4;
```

Wrong CUDA 12.5 (`asm/cuda-12.5.ptx`):

```ptx
selp.b32  %r52, %r44, 2039404531, %p3;
selp.b32  %r53, 0, %r51, %p3;
shl.b32   %r54, %r53, 2;
xor.b32   %r56, %r54, -10;               ; (x^9);not folded to (x^~9)
add.s32   %r4, %r53, %r56;               ; v_b + ((v_b<<2) ^ ~9)  -- ~v_a gone
setp.lt.u32 %p4, %nctaid.y, %r4;
```

Since `x ^ ~9 = ~(x ^ 9) = -(x ^ 9) - 1`, the 12.5 expression equals the 13.0
expression only when `v_a == 0`. cicc propagated a bogus zero for the
`selp`-produced value `v_a` and dropped the whole `~v_a` addend.

## Root cause

A bogus zero/constant-propagation error in the CUDA 12.5 NVVM/cicc
optimizer: a runtime `selp` result is treated as known-zero and the term
derived from it is removed from an add/sub chain. Fixed in 12.6. Same
component family as `miscompile-all-cicc-mulnarrow0` (12.5/12.6 cicc), but a
distinct transformation. Versions before 12.5 are not installed on this host
and remain unverified.
