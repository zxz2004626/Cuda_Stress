# Investigation: second ptxas O1+ divergence class -- over-MSB bfe/bfi fields

case:      inline_ptx__ebf45243624dc5d6
strategy:  inline_ptx  (n_instr=120, n_regs32=12, n_regs64=6, n_preds=5)
config_seed: 20260810   test_seed: 777   kernel_seed: 8337606540844322850
GPU:       NVIDIA GeForce RTX 2050 (sm_86), driver UMD 12.1

## Symptom
Same kernel: -O0 matches the CPU oracle on all 128 words (all runnable toolchains);
-O1/-O2/-O3 produce deterministic but different output on cuda-11.8/12.1/12.6.
cuda-11.3 stays correct at all levels.

## Root cause (isolated)
The divergence is essential to the over-MSB bfe/bfi instructions in this program
(6 instructions with pos+wid > 32).  Replacing ONLY those over-MSB bfe/bfi with movs
makes every O level agree.  The same over-MSB bfe/bfi computed in an isolated kernel is
correct at O0..O3, so (like the prmt@37 case) this is ptxas's context-sensitive
optimization of over-MSB bfe/bfi inside a large inline-asm block.

## Why this is NOT testcase UB
PTX bfe/bfi take position/length in [0,255]; a field extending past the register MSB is
WELL-DEFINED -- bfe.u32 zero-extends the missing high bits, bfi.b32 simply drops the
excess (verified on-GPU: bfe(a,24,16), bfe(a,32,8), bfi(a,b,24,16), bfi(a,b,255,255),
all correct; the Python mirrors agree 64/64 across pos 0..252 / wid 1..32).
The generator now samples this full legal over-MSB region.

## Note
This case was only reachable after widening the bfe/bfi generation to the full legal
range.  The earlier, more conservative pos+wid<=32 constraint had been hiding these
additional ptxas O1+ miscompilations (the case was mislabeled 'bfe/bfi UB' at first;
that claim was wrong and has been removed).
