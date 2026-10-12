# C67: ptxas -O1+ constant-folding bug since 11.8 (u64 division chain, off-by-2), unfixed

Self-contained, stable, UB-free reduced reproducer distilled from the
Cuda_Stress differential run (trigger-matrix class **C67**).

## Trigger matrix (archived, sm_86, CUDA 11.1..13.3 × ptxas O0/O1/O2)

`11.8/O1-O2 + 12.2/O1-O2 + 12.5/O1-O2 + 12.8/O1-O2 + 13.0/O1-O2 + 13.3/O1-O2`

The archived matrix groups the 27 (version × ptxas -O) builds by output
checksum; the listed cells are the minority (deviating) builds.

## Root cause (confirmed, deep-dive analysis of `test.cu`)

**结论：源码无 UB。ptxas 对"按 2²⁴ 缩放的 8 位符号比较"的降级错误：
PTX（三级 `-Xptxas` 逐字节一致，且 11.5 与 12.6 的 PTX 也仅差版本头）把
`v_25 > (s8)-61` 正确降级为 `setp.gt.s32 %p3, %r27, -1023410176`
（`%r27 = v25 << 24`，`-1023410176 = -61 << 24`），但 ptxas 11.5（全部级别）
与 13.3 -O0 生成的 SASS 对部分 lane 给出与语义相反的谓词。**

### 实测（RTX 3090, sm_86，重测确认）

- 标准/源语义（Python 模拟，逐位）：`6136588885113614127`
  （= 12.6 `-O3`、13.3 `-Xptxas=-O1/-O2/-O3`）
- 错误：`16613416081537034649`
  （= 11.5 `-O0/-O3/-G`、13.3 `-Xptxas=-O0`）
- 逐线程 diff：全部 16384 线程差 bit1（ballot bit1，即 lane 1 谓词经 `+2` 的进位）；
  反推 ballot 显示差异不止 lane 1（如 warp4 的 A^B = 0xFFE，lane 1–11 翻转）——
  标准语义中这些 lane 的 `v_25` 均为 16 的倍数（如 32），
  `32 > -61` 应为 true，错误构建给出 false。

### PTX / SASS 证据

- PTX（正确）：`setp.gt.s32 %p3, %r27, -1023410176` + `vote.sync.ballot.b32`；
- SASS：11.5 与 13.3-O0 的谓词结果与语义相反（12.6/13.3 的 -O1+ 正确）——
  属 ptxas "小类型缩放比较" 降级家族（与 C32/C50 的 `neg.s16` 截断丢弃、
  已归档的 `u16-signext0/1` 同环节），但触发条件不同（无需 INT16_MIN）。

---

## Root cause (preliminary, from archived trigger matrix)

ptxas -O1+ 优化 bug（11.8 引入，未修）：lane 无关常量（u64 除法倒数链）算错，全 lane 恒差 +2。与 umul64hi（sub.s64 高位字，-O1+、11.1 起）机制不同。

Evidence (cross-ptxas sweep, see `rc_sweep` reference below): the failing-side
PTX was assembled with every toolkit's ptxas and executed; the per-version /
per-opt checksum matrix pins the defect to the compiler layer and window shown
above. The reduced core is deterministic across repeated runs and clean under
compute-sanitizer memcheck / initcheck / racecheck.


Status: **unfixed** as of CUDA 13.3.

## Minimal core (`min.cu`, 2026-09-21 reduction)

`test.cu` was further reduced to `min.cu` (1 block × 8 threads, output =
`(u16)` of the u64 division chain alone; no input buffer, no `n` dependency).

```bash
make min            # builds min_O0 / min_O1 with CUDA 13.3 and prints both checksums
```

The isolated core is the lane-uniform division chain

```
q = ((unsigned long long)var_1 * 9223372036854775807ull) / ((~var_1) | 1ull);
// var_1 = 16694769789140031405:
// num = 0x985043545762a853, den = 0x185043545762a853  ->  q = 6 (true)
```

At ptxas -O1/-O2/-O3 (CUDA 11.8..13.3) the reciprocal-based division sequence
folds `q` to **0**; ptxas -O0 and CUDA 11.5 (all levels) compute 6. In
`test.cu` the quotient feeds `| 0x6461bdc4`, so `(u16)v_16` is 0xbdc6 (correct)
vs 0xbdc4 (wrong) — the archived per-lane ±2 signature (measured directly:
`v_16 = 1684127174` at -O0 vs `1684127172` at -O1).

Measured for `min.cu` (RTX 3090, sm_86, this exact file — the bug is
context-sensitive: deleting the dead `v_0/v_18/v_25` chain, the `__ballot_sync`
statement, or replacing the ballot by a constant each kills the trigger):

| toolkit | ptxas | checksum | verdict |
| --- | --- | --- | --- |
| 11.8 / 12.2 / 12.5 / 12.8 / 13.0 / 13.1 / 13.2 / 13.3 | -O0 | `16938453490336023656` | CORRECT |
| 11.8 / 12.2 / 12.5 / 12.8 / 13.0 / 13.1 / 13.2 / 13.3 | -O1/-O2/-O3 | `16938317252721529361` | WRONG |
| 11.5 (negative control) | -O0/-O1/-O2 | `16938453490336023656` | CORRECT |

compute-sanitizer memcheck / initcheck / racecheck on the -O1 build: 0 errors.
The kernel is UB-free (single u64 division by a non-zero runtime value).

Relationship to the two root-cause paragraphs below: `min.cu` isolates the
**u64 division-chain misfold** (metadata `root_cause`) and removes the
v_25-comparison deviation discussed in "Root cause (confirmed)" entirely —
min.cu's output depends only on `q`, whose correct value 6 is trivially
verifiable, so this reduction pins the division bug unambiguously. The
v_25/ballot chain is retained only as scheduling context (its value never
reaches the output).

## Files

| file | description |
| --- | --- |
| `min.cu` | Minimal core reproducer (this reduction; see section above). |
| `test.cu` | Self-contained minimal reproducer (device kernel + host `main` that prints `checksum=<u64>`). |
| `Makefile` | Builds the BAD and GOOD configurations and checks both checksums (`make min` builds min.cu). |
| `metadata.yaml` | Machine-readable summary (trigger, checksums, provenance). |

## Usage

```bash
cd /export/ssd1/xzhanglb/llm-template/Cuda_Stress/poc/C67
make            # builds with CUDA 13.3 ptxas -O0/-O1/-O2, prints each checksum
make clean
```

Use a different toolkit:

```bash
make NVCC=/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.0/bin/nvcc
```

## Expected result (measured on RTX 3090, sm_86)

| toolkit | -Xptxas | verdict | checksum |
| --- | --- | --- | --- |
| CUDA 11.8 | -O0 | CORRECT | 16613416081537034649 |
| CUDA 11.8 | -O1 | WRONG | 6136588885113614127 |
| CUDA 11.8 | -O2 | WRONG | 6136588885113614127 |
| CUDA 11.8 | -O3 | WRONG | 6136588885113614127 |

- source-correct checksum (GOOD side): `16613416081537034649`
- miscompiled checksum (BAD side): `6136588885113614127`

## Provenance

- differential class: `C67` of run set
  Cuda_Stress(2026-09-11) / Cuda_Stress_ds(2026-09-14) / Cuda_Stress_glm(2026-09-13) /
  Cuda_Stress_glmmax(2026-09-15) / Cuda_Stress_gpt(2026-09-16)
- reduced core: `reduce_out/C67` (`textreduced.cu`), produced by
  `generator_ir.reduce` with UB gates; verification: `ub_check.json`, cross-ptxas
  evidence: `rc_sweep.json` in the same directory.
- 27-cell faithful re-sweep of the class representative: `trigger_matrix_analysis/faithful_sweep.jsonl`.
