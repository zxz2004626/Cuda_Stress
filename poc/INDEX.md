# CUDA compiler differential testcase index

Neutral-ID reorganization of `poc/`. Each directory carries its own `metadata.yaml`;
this index is a one-line-per-case summary. Trigger ranges are the **sm_86** sweep
(CUDA 11.1..13.3, the only arch runnable natively on this host, RTX 3090); `sm_75`
ranges are copied from each case's original README and marked as such.

| ID | class | opt | sm_86 range | correct → buggy | root cause (short) |
| --- | --- | --- | --- | --- | --- |
| `deadloop-O0-chunkloop-init0` | deadloop | O0 | not repro (sm_86) | None → non-terminating | ptxas -O0 places the -64 induction-var init off the first-entry path o |
| `deadloop-O0-chunkloop-init1` | deadloop | O0 | not repro (sm_86) | None → non-terminating | Same family as chunkloop-init0: second inlined helper_0 call has a pat |
| `hang-O0-prefixscan` | hang | O0,O1+ | 11.1..13.3 (unfixed) | None → non-terminating | ptxas backend SASS control-flow/scheduling error on a small bounded lo |
| `miscompile-O1-xorpred` | miscompile | O1 | not repro (sm_86) | 1 → 0 | ptxas drops/inverts a lowest-bit predicate while reordering the mixed  |
| `miscompile-O1plus-umul64hi` | miscompile | O1,O2,O3 | 11.1..13.3 (unfixed) | 0 → 1027807174 | ptxas fixes the high word of a sub.s64 0-0 result to -1 (IADD3.X ... - |
| `miscompile-all-mulwide32` | miscompile | O0,O1,O2,O3 | 11.2..11.5, fixed 11.6 | 1073741824 → 3221225472 | (unsigned long long)(signed int)<<31 lowered to mul.wide.s32 var |
| `miscompile-O1-prmt-rz0` | miscompile | O1 | 11.2..11.6, fixed 11.7 | 2654448107 → 2654448106 | ptxas -O1 writes the __byte_perm result to RZ and drops the and+setp.n |
| `miscompile-O1-prmt-rz1` | miscompile | O1,O2 | 11.1..11.6, fixed 11.7 | 197 → 229 | Same PRMT->RZ family: __byte_perm(...)!=0 ? 49:0 reuses the stale 0<th |
| `miscompile-O2-atomic-xor` | miscompile | O2,O3 | 11.1..11.6, fixed 11.7 | 2419750549 → 2419750548 | ptxas -O2 unrolls the 3-iter loop and reuses the wrong live register a |
| `miscompile-O1-ffsll` | miscompile | O1 | 11.7..12.1, fixed 12.2 | 5850102576 → 5850102547 | ptxas -O1 uniform-register path uses +gridDim.x instead of -gridDim.x  |
| `miscompile-all-mulwide32-frontend` | miscompile | O0,O1,O2,O3 | 11.1..11.5, fixed 11.6 | 0 → 32 | NVVM front-end lowers (unsigned long long)(int)n<<31 to mul.wide.s32 n |
| `miscompile-O1plus-rem-prenot` | miscompile | O1,O2,O3 | 11.6..12.1, fixed 12.2 | 1 → 0 | ptxas -O1 folds the and+setp.eq zero-divisor guard of rem.s32 using th |
| `miscompile-O1plus-div-prenot` | miscompile | O1,O2,O3 | 11.6..12.1, fixed 12.2 | 65520 → 3 | ptxas -O1 tests high32(x)!=0 instead of high32(~x)!=0 for the div.u64  |
| `hang-O0-threshold` | hang | O0 | 12.0..13.3 (unfixed) | 0 → over-threshold (30s) | finite stress loop (strided global loads + atomic + __syncthreads + ne |
| `miscompile-O1plus-u16-signext0` | miscompile | O1,O2 | 11.2..13.2, fixed 13.3 | 65407 → 127 | ptxas omits the second PRMT re-sign-extension after neg.s16 |
| `miscompile-O1plus-u16-signext1` | miscompile | O1,O2 | 11.2..13.2, fixed 13.3 | 4294967168 → 128 | Same u16 sign-extension bug as signext0 |
| `miscompile-all-cicc-mulnarrow0` | miscompile | O0,O1,O2 | <=12.5..12.6, fixed 12.8 | 0 → 1 | cicc moves multiply across char narrowing: 3*(char)(c-129) -> 3*c+381 |
| `miscompile-all-cicc-dropaddend0` | miscompile | O0,O1,O2 | <=12.5, fixed 12.6 | 15181709146300232064 → 13004433893898885072 | cicc drops nonzero selp addend ~v_a via bogus zero propagation |

## Notes

- `deadloop-*` and `miscompile-O1-xorpred` are **sm_75-only**: they do not reproduce
  at sm_86 in 11.1..13.3 (see each `metadata.yaml` for the sm_75 range).
- `miscompile-all-mulwide32-frontend`: the *correct* side is the **later** toolchain
  (11.6+ matches source semantics); 11.1..11.5 emit the buggy front-end value.
- `hang-O0-threshold`: threshold-style, **not** infinite — O0 exceeds the 30s harness
  threshold at 12.0, 12.3..12.5, 13.1..13.3; O1 mostly completes <22s.
- `unverifiable_versions` on this host: 10.0, 10.2, 11.0 (sm_86 needs CUDA >= 11.1).
- `miscompile-all-cicc-*`: **cicc/NVVM front-end** bugs, not ptxas — wrong at every
  `-Xptxas` level; proven by PTX-JIT (`-arch=compute_86`, driver JIT reproduces the
  wrong value) and by `-G` (optimizer off) being correct on the affected toolchains.
  Sweeps covered 12.5, 12.6, 13.0..13.3 natively; mulnarrow additionally verified
  fixed on docker 12.8.2/12.9.2 (CUDA 12.7 does not exist). `introduced` is
  "<=12.5" for both; 11.x..12.4 not installed on this host.
- `miscompile-all-cicc-mulnarrow0` needs the dead `__ldg(i);` statement to trigger;
  original case: `generator_ir__350e41ea3b05302b` (run-20260829-011020-1bbc).
- `miscompile-all-cicc-dropaddend0`: the IR-level reducer output for the original case
  (`generator_ir__0948c44ea19f0120`) is a UB artifact (empty non-void helper_1) and is
  NOT the mechanism in this PoC; this PoC uses a source-level reduction of the original.
